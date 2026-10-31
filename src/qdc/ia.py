"""Phase 4b : un LLM tranche les cas ambigus du dedoublonnage.

Pourquoi un LLM ICI et pas partout ?
- Les regles traitent 95 % des paires en quelques secondes, gratuitement.
- Les cas ambigus demandent du "bon sens" : « A. Leduc » et « Aurelie Leduc » au meme
  code postal, c'est probablement la meme personne ; « Colette » et un prenom vide, on ne sait pas.
- Le LLM ne decide pas seul : s'il n'est pas assez sur, le cas part en revue humaine.

Points d'attention (a savoir expliquer) :
- COUT / LIMITES : on traite par lots de 10 paires et on met les reponses en cache.
- FIABILITE : on mesure l'exactitude du LLM contre la verite sur un echantillon AVANT de
  l'appliquer a tout.
- RGPD : ici les donnees sont fictives. Avec de vraies donnees personnelles, il faut un
  fournisseur sous contrat (DPA), heberge en UE, ou un modele interne.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import duckdb
import polars as pl
from rapidfuzz.distance import JaroWinkler

from qdc import regles

# Modele different de celui qui redige les reponses de Logi (gpt-oss-120b) : chez Groq, les
# quotas sont PAR MODELE. Un gros traitement ici ne peut donc pas couper la demo Logi.
MODELE = os.environ.get("QDC_MODELE", "openai/gpt-oss-20b")
VERSION_PROMPT = "v3"  # changer de prompt = nouvelles questions : le cache ne doit pas les confondre
TAILLE_LOT = 5         # au-dela, le modele melange parfois les paires d'un meme lot
# Deux seuils distincts, fixes par l'analyse de calibration (voir `calibration` ci-dessous) :
# une fausse fusion est grave -> on exige plus de confiance pour fusionner que pour separer.
CONFIANCE_FUSION = 0.9      # « meme personne » accepte seulement au-dela
CONFIANCE_SEPARATION = 0.8  # « personnes differentes » accepte au-dela
# En dessous de ces seuils, le cas part en revue humaine.
CHAMPS = ["source", "civilite", "prenom", "nom", "email", "telephone", "adresse",
          "code_postal", "ville", "date_naissance"]

PROMPT_SYSTEME = """Tu es un expert en qualite des donnees clients (gestion de referentiel client).
On te donne des paires de fiches clients issues de systemes differents (CRM, boutique en ligne,
service client). Pour chaque paire, decide si les deux fiches designent la MEME personne.

Pour chaque paire, tu recois les deux fiches ET une comparaison champ par champ deja calculee
(identique / proche / different / inconnu). Appuie-toi sur cette comparaison : elle est fiable.

Connaissances utiles :
- "inconnu" (champ absent) n'est ni un indice pour, ni contre.
- Une initiale (« A. ») est compatible avec un prenom commencant par cette lettre.
- Les fautes de frappe sont frequentes (Dupont / Dupond).
- Une meme personne peut avoir un e-mail perso et un e-mail pro : compare la partie avant le @
  (« jean.martin@gmail.com » et « jmartin@entreprise.fr » sont compatibles).
- Une meme personne peut avoir change de numero ou d'adresse (demenagement).
- PIEGE : deux personnes d'un meme foyer partagent souvent le nom, l'adresse et le telephone FIXE
  (+331 a +335), mais ont des prenoms, civilites ou dates de naissance differents.
  Un mobile (+336, +337) est personnel.
- Un nom de famille et un code postal communs ne suffisent PAS : il y a beaucoup d'homonymes.
- Si "prenom_et_nom_inverses" vaut "oui", le prenom d'une fiche a ete saisi dans le champ nom :
  compare alors le prenom de l'une avec le nom de l'autre.
- Des dates de naissance differentes signifient deux personnes differentes.

Regle de prudence : une fausse fusion (melanger deux personnes) est bien plus grave qu'un doublon
manque. En cas de doute, reponds meme_personne=false avec une confiance basse.
Ne donne une confiance >= 0.9 que si plusieurs indices INDEPENDANTS concordent.

Reponds UNIQUEMENT en JSON, au format :
{"decisions": [{"id": "<id de la paire>", "nom_fiche_1": "<recopie le champ nom (nom de famille) de la fiche 1>",
                "meme_personne": true|false, "confiance": <0 a 1>,
                "raison": "<une phrase courte en francais>"}]}"""


# --- Preparation des questions ---------------------------------------------------------
def comparer(v1: str | None, v2: str | None) -> str:
    if v1 is None or v2 is None:
        return "inconnu"
    a, b = (pl.select(regles.norm_ville(pl.lit(x))).item() for x in (v1, v2))
    if a == b:
        return "identique"
    return "proche" if JaroWinkler.normalized_similarity(a, b) >= 0.88 else "different"


def _fiche(r: dict, suffixe: str) -> dict:
    return {c: (str(r[c + suffixe]) if r[c + suffixe] is not None else None) for c in CHAMPS}


def cle_cache(f1: dict, f2: dict) -> str:
    """Empreinte du CONTENU de la question (et non des identifiants) : si les donnees changent,
    les identifiants peuvent designer d'autres personnes ; l'empreinte, elle, change aussi."""
    contenu = json.dumps([VERSION_PROMPT, MODELE, f1, f2], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(contenu.encode()).hexdigest()[:32]


def preparer_lot(paires: pl.DataFrame, silver: pl.DataFrame) -> list[dict]:
    """Associe a chaque paire ambigue les deux fiches et leur comparaison champ par champ."""
    s = silver.select(["record_id"] + CHAMPS)
    ab = paires.join(s, left_on="id_a", right_on="record_id").join(s, left_on="id_b", right_on="record_id", suffix="_b")
    questions = []
    for r in ab.iter_rows(named=True):
        f1, f2 = _fiche(r, ""), _fiche(r, "_b")
        comparaison = {c: comparer(f1[c], f2[c]) for c in CHAMPS if c != "source"}
        # Prenom et nom inverses d'une fiche a l'autre (« Julien Patrick » / « Patrick Julien »)
        inverses = (comparer(f1["prenom"], f2["nom"]) in ("identique", "proche")
                    and comparer(f1["nom"], f2["prenom"]) in ("identique", "proche"))
        comparaison["prenom_et_nom_inverses"] = "oui" if inverses else "non"
        questions.append({"id": f"{r['id_a']}|{r['id_b']}", "fiche_1": f1, "fiche_2": f2,
                          "comparaison": comparaison, "_cle": cle_cache(f1, f2)})
    return questions


def appeler_groq(messages: list[dict]) -> str:
    """Appel reel a Groq. La cle est lue dans l'environnement (fichier .env), jamais dans le code."""
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    options = {"reasoning_effort": "low"} if MODELE.startswith("openai/gpt-oss") else {}
    resp = client.chat.completions.create(
        model=MODELE, messages=messages, temperature=0, max_tokens=3000,
        response_format={"type": "json_object"}, **options,
    )
    return resp.choices[0].message.content or "{}"


def _meme_nom(recopie, attendu) -> bool:
    """Le LLM a-t-il recopie le bon nom ? Tolerant : « T. Menard », « MENARD » ou « Menard »
    conviennent pour le nom « Menard ». Un autre nom de famille = il confond les paires."""
    if attendu is None:
        return True  # rien a verifier
    if not isinstance(recopie, str) or not recopie.strip():
        return False
    cle = lambda x: pl.select(regles.norm_ville(pl.lit(x))).item() or ""  # noqa: E731
    r, a = cle(recopie), cle(attendu)
    return a in r or any(JaroWinkler.normalized_similarity(mot, a) >= 0.88 for mot in r.replace(".", " ").split())


class QuotaJournalierAtteint(Exception):
    """Plus de tokens pour aujourd'hui : inutile d'insister, on reprendra demain (grace au cache)."""


def _attente_conseillee(message: str, defaut: float) -> float:
    """Groq indique combien attendre (« Please try again in 7.5s » ou « in 1m2.3s »)."""
    import re
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else defaut


def juger(lot: list[dict], appeler: Callable[[list[dict]], str] = appeler_groq, essais: int = 4) -> list[dict]:
    """Envoie un lot au LLM. Garde-fous :
    - on ignore les identifiants inventes et les reponses mal formees ;
    - le LLM doit RECOPIER le nom de famille de la fiche 1 : s'il se trompe, c'est qu'il confond
      les paires du lot, et on jette cette decision (plutot qu'une decision sur la mauvaise paire).
    On ne relance le lot QUE si l'appel a echoue (erreur, JSON illisible) : une reponse valide mais
    partielle est gardee telle quelle, sinon on paierait plusieurs fois le meme lot."""
    envoi = [{k: v for k, v in q.items() if not k.startswith("_")} for q in lot]
    messages = [{"role": "system", "content": PROMPT_SYSTEME},
                {"role": "user", "content": json.dumps({"paires": envoi}, ensure_ascii=False)}]
    par_id = {q["id"]: q for q in lot}
    for essai in range(essais):
        try:
            brut = json.loads(appeler(messages))
        except json.JSONDecodeError:
            print(f"      essai {essai + 1} : reponse illisible, nouvelle tentative")
            time.sleep(5)
            continue
        except Exception as e:  # limite de debit, erreur du service...
            msg = str(e)
            if "tokens per day" in msg or "(TPD)" in msg:
                raise QuotaJournalierAtteint(msg[:200]) from e
            attente = _attente_conseillee(msg, 20 * (essai + 1))
            print(f"      essai {essai + 1} echoue : {type(e).__name__}, nouvelle tentative dans {attente:.0f} s")
            time.sleep(attente)
            continue
        decisions = brut.get("decisions", []) if isinstance(brut, dict) else []
        decisions = [d for d in decisions if isinstance(d, dict) and d.get("id") in par_id
                     and isinstance(d.get("meme_personne"), bool)]
        valides = [d for d in decisions if _meme_nom(d.get("nom_fiche_1"), par_id[d["id"]]["fiche_1"]["nom"])]
        if len(valides) < len(decisions):
            print(f"      {len(decisions) - len(valides)} decision(s) rejetee(s) : nom recopie incorrect")
        return [appliquer_verrous(d, par_id[d["id"]]) for d in valides]
    return []


def appliquer_verrous(d: dict, question: dict) -> dict:
    """Regles IMPERATIVES, appliquees par le code et non demandees au modele : le LLM peut
    ignorer une consigne du prompt, pas une ligne de Python."""
    if d["meme_personne"] and question.get("comparaison", {}).get("date_naissance") == "different":
        return d | {"meme_personne": False, "confiance": 1.0,
                    "raison": "Verrou : dates de naissance differentes (avis IA ecarte : " + str(d.get("raison", "")) + ")"}
    return d



# --- Cache : ne jamais payer deux fois la meme question --------------------------------
def _cache(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS cache_ia (
        cle VARCHAR PRIMARY KEY, meme_personne BOOLEAN, confiance DOUBLE, raison VARCHAR,
        modele VARCHAR, version_prompt VARCHAR, decide_le TIMESTAMP DEFAULT current_timestamp)""")


def lire_cache(base: Path) -> pl.DataFrame:
    with duckdb.connect(str(base)) as con:
        _cache(con)
        return con.sql("SELECT cle, meme_personne, confiance, raison FROM cache_ia").pl()


def enregistrer(base: Path, decisions: list[dict], cles: dict[str, str]) -> None:
    with duckdb.connect(str(base)) as con:
        _cache(con)
        for d in decisions:
            try:
                confiance = min(max(float(d.get("confiance", 0)), 0.0), 1.0)
            except (TypeError, ValueError):
                confiance = 0.0
            con.execute("INSERT OR REPLACE INTO cache_ia (cle, meme_personne, confiance, raison, modele, version_prompt) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [cles[d["id"]], d["meme_personne"], confiance, str(d.get("raison", "")), MODELE, VERSION_PROMPT])


def traiter(paires_ambigues: pl.DataFrame, silver: pl.DataFrame, base: Path,
            appeler: Callable[[list[dict]], str] = appeler_groq, pause: float = 8.0) -> pl.DataFrame:
    """Juge les paires pas encore en cache, lot par lot. Renvoie les decisions de TOUTES les paires demandees."""
    questions = preparer_lot(paires_ambigues, silver)
    deja = set(lire_cache(base)["cle"].to_list())
    a_faire = [q for q in questions if q["_cle"] not in deja]
    print(f"  {len(a_faire)} paires a juger ({len(questions) - len(a_faire)} deja en cache)")
    n_lots = -(-len(a_faire) // TAILLE_LOT)
    for i in range(0, len(a_faire), TAILLE_LOT):
        lot = a_faire[i:i + TAILLE_LOT]
        try:
            decisions = juger(lot, appeler)
        except QuotaJournalierAtteint:
            print("\n  QUOTA JOURNALIER ATTEINT : arret propre. Relance demain la meme commande,")
            print("  les paires deja jugees sont en cache et ne seront pas renvoyees.")
            break
        enregistrer(base, decisions, {q["id"]: q["_cle"] for q in lot})
        print(f"    lot {i // TAILLE_LOT + 1}/{n_lots} : {len(decisions)}/{len(lot)} decisions")
        time.sleep(pause)  # respecter les limites de debit du palier gratuit
    index = pl.DataFrame({"id": [q["id"] for q in questions], "cle": [q["_cle"] for q in questions]})
    return index.join(lire_cache(base), on="cle").drop("cle")


# --- Evaluation : l'IA aide-t-elle vraiment ? ----------------------------------------------
def evaluer_ia(decisions: pl.DataFrame, verite: pl.DataFrame) -> dict:
    pid = verite.select("record_id", "person_id")
    d = (decisions.with_columns(pl.col("id").str.split("|").list.to_struct(fields=["id_a", "id_b"])).unnest("id")
         .join(pid, left_on="id_a", right_on="record_id").join(pid, left_on="id_b", right_on="record_id", suffix="_b")
         .with_columns(vrai=pl.col("person_id") == pl.col("person_id_b"),
                       sur=(pl.col("meme_personne") & (pl.col("confiance") >= CONFIANCE_FUSION))
                       | (~pl.col("meme_personne") & (pl.col("confiance") >= CONFIANCE_SEPARATION))))
    surs = d.filter("sur")
    vp = surs.filter(pl.col("meme_personne") & pl.col("vrai")).height
    fp = surs.filter(pl.col("meme_personne") & ~pl.col("vrai")).height
    fn = surs.filter(~pl.col("meme_personne") & pl.col("vrai")).height
    return {
        "jugees": d.height,
        "exactitude_globale": d.filter(pl.col("meme_personne") == pl.col("vrai")).height / max(d.height, 1),
        "decidees_seules": surs.height,
        "exactitude_si_sure": surs.filter(pl.col("meme_personne") == pl.col("vrai")).height / max(surs.height, 1),
        "precision_fusions": vp / max(vp + fp, 1),
        "doublons_rates": fn,
        "revue_humaine": d.height - surs.height,
        "erreurs": d.filter(pl.col("sur") & (pl.col("meme_personne") != pl.col("vrai"))),
        # Calibration : quand l'IA annonce telle confiance, a-t-elle raison aussi souvent ?
        "calibration": d.group_by(
            pl.when(pl.col("meme_personne")).then(pl.lit("meme")).otherwise(pl.lit("different")).alias("reponse"),
            (pl.col("confiance") * 20).floor().truediv(20).alias("confiance"),
        ).agg(n=pl.len(), exactitude=(pl.col("meme_personne") == pl.col("vrai")).mean())
        .sort("reponse", "confiance", descending=[False, True]),
    }


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    from qdc.dedoublonnage import evaluer, regrouper

    load_dotenv()  # lit GROQ_API_KEY dans le fichier .env (jamais versionne)
    parser = argparse.ArgumentParser(description="Le LLM tranche les cas ambigus.")
    parser.add_argument("--echantillon", type=int, default=300,
                        help="nombre de cas ambigus a juger (0 = tous)")
    parser.add_argument("--modele", default=None,
                        help="modele Groq a utiliser (par defaut : QDC_MODELE ou openai/gpt-oss-20b)")
    parser.add_argument("--pause", type=float, default=8.0,
                        help="secondes entre deux appels (palier gratuit : rester sous la limite par minute)")
    args = parser.parse_args()
    if args.modele:
        MODELE = args.modele  # noqa: F811 (le modele fait partie de la cle du cache)
    print(f"  modele : {MODELE}")

    base = Path("data/qdc.duckdb")
    with duckdb.connect(str(base)) as con:
        silver = con.sql("SELECT * FROM silver_clients").pl()
        paires = con.sql("SELECT * FROM paires").pl()
    ambigues = paires.filter(pl.col("decision") == "ambigu")
    if args.echantillon:
        ambigues = ambigues.sample(min(args.echantillon, ambigues.height), seed=42)

    decisions = traiter(ambigues.select("id_a", "id_b"), silver, base, pause=args.pause)
    if decisions.is_empty():
        raise SystemExit("Aucune decision disponible pour l'instant.")

    verite = pl.read_csv("data/verite/verite.csv")
    ev = evaluer_ia(decisions, verite)
    print(f"\n  L'IA A JUGE {ev['jugees']} CAS AMBIGUS")
    print(f"    exactitude globale            {100 * ev['exactitude_globale']:5.1f} %")
    print(f"    decides seuls                 {ev['decidees_seules']:>4}   exactitude {100 * ev['exactitude_si_sure']:5.1f} %"
          f"   (fusion si confiance >= {CONFIANCE_FUSION}, separation si >= {CONFIANCE_SEPARATION})")
    print(f"    precision des fusions IA      {100 * ev['precision_fusions']:5.1f} %")
    print(f"    envoyes en revue humaine      {ev['revue_humaine']:>4}")
    print("\n  CALIBRATION : l'IA a-t-elle raison quand elle se dit sure ?")
    print("    reponse     confiance   cas   exactitude")
    for r in ev["calibration"].iter_rows(named=True):
        print(f"    {r['reponse']:<10}  >= {r['confiance']:.2f}   {r['n']:>4}   {100 * r['exactitude']:5.1f} %")
    print("\n  ERREURS DE L'IA (quand elle etait sure d'elle)")
    for r in ev["erreurs"].head(5).iter_rows(named=True):
        print(f"    {r['id_a']} | {r['id_b']}  dit {'meme' if r['meme_personne'] else 'different'} "
              f"({r['confiance']:.2f}) : {r['raison']}")

    if not args.echantillon:  # sur l'ensemble : impact sur la qualite globale
        surs = decisions.filter((pl.col("confiance") >= CONFIANCE_FUSION) & pl.col("meme_personne"))
        liens_ia = surs.with_columns(pl.col("id").str.split("|").list.to_struct(fields=["id_a", "id_b"])).unnest("id")
        liens = pl.concat([paires.filter(pl.col("decision") == "doublon").select("id_a", "id_b"),
                           liens_ia.select("id_a", "id_b")])
        e = evaluer(regrouper(silver["record_id"], liens), verite)
        print(f"\n  AVEC L'IA : precision {100 * e.precision:.1f} %  rappel {100 * e.rappel:.1f} %  F1 {100 * e.f1:.1f} %")
