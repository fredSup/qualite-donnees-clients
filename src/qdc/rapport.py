"""Phase 6 : le rapport d'audit, ecrit pour un responsable marketing (pas pour un data engineer).

Principe : LE CODE CALCULE, L'IA REDIGE.
1. On extrait de la base tous les faits chiffres (volumes, scores avant/apres, problemes par source).
2. Le LLM les met en forme et propose des recommandations, avec l'interdiction d'inventer un chiffre.
3. Garde-fou : le programme verifie que CHAQUE nombre du rapport figure bien dans les faits.
   Un nombre introuvable est signale en tete du rapport.
4. Sans cle API (ou si l'appel echoue), un rapport est quand meme produit a partir d'un modele
   de texte fixe : l'IA ameliore l'outil, elle n'en est pas une dependance.

Le rapport ne cite AUCUNE mesure calculee avec la "verite" (precision, rappel) : en entreprise,
la verite n'existe pas. On ne rapporte que ce qu'un vrai client pourrait verifier.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from qdc import diagnostic, fusion
from qdc.diagnostic import FRAICHEUR_ANS

NOMS_SOURCES = {"crm": "CRM", "boutique": "boutique en ligne", "support": "service client"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def nb(x: int) -> str:
    return f"{x:,}".replace(",", " ")




def collecter_faits(base: Path, aujourd_hui: date | None = None) -> dict:
    """Tous les chiffres du rapport, deja formates a la francaise (« 79,1 », « 10 407 »).

    Les diagnostics sont RECALCULES ici a partir des donnees actuelles, et non lus dans une table
    stockee : une table de resultats peut etre perimee si on a regenere les donnees sans
    relancer l'etape qui l'ecrit (bug rencontre pendant le projet)."""
    aujourd_hui = aujourd_hui or date.today()
    with duckdb.connect(str(base), read_only=True) as con:
        q = lambda sql: con.sql(sql).pl()  # noqa: E731
        bronze = q("SELECT * FROM bronze_clients")
        # Journal du nettoyage PAR SOURCE : sans ce detail, l'IA attribuait les chiffres au hasard
        journal = q("""SELECT source, 'correction' AS type, op, count(*) AS n
                       FROM (SELECT source, unnest(corrections) AS op FROM silver_clients) GROUP BY ALL
                       UNION ALL
                       SELECT source, 'anomalie', op, count(*)
                       FROM (SELECT source, unnest(anomalies) AS op FROM silver_clients) GROUP BY ALL""")
        paires = q("SELECT decision, count(*) AS n FROM paires GROUP BY decision")
        gold = q("SELECT * FROM gold_clients")
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        if "cache_ia" in tables:
            silver = q("SELECT * FROM silver_clients")
            ambigues = q("SELECT id_a, id_b FROM paires WHERE decision = 'ambigu'")
    # Cas ambigus REELLEMENT juges par l'IA : paires actuelles x version actuelle du prompt
    # (le cache contient aussi d'anciennes reponses, qu'il ne faut pas compter)
    n_ia = 0
    if "cache_ia" in tables:
        from qdc import ia
        cles = {q_["_cle"] for q_ in ia.preparer_lot(ambigues, silver)}
        n_ia = ia.lire_cache(base).filter(pl.col("cle").is_in(list(cles))).height

    diag_avant = diagnostic.diagnostiquer(bronze, aujourd_hui)
    df_g, applicable = fusion.pour_diagnostic(gold)
    diag_apres = diagnostic.diagnostiquer(df_g, aujourd_hui, applicable=applicable)
    avant = diag_avant.details
    s_avant, s_apres = diag_avant.score_global / 100, diag_apres.score_global / 100
    d_avant = dict(diag_avant.resume.select("dimension", "taux").iter_rows())
    d_apres = dict(diag_apres.resume.select("dimension", "taux").iter_rows())
    par_source = {}
    for src, part in avant.group_by("source"):
        src = src[0]
        if src not in NOMS_SOURCES:
            continue
        faibles = part.filter((pl.col("taux") < 0.95) & (pl.col("champ") != "doublons internes")).sort("taux")
        j = journal.filter(pl.col("source") == src).sort("n", descending=True)
        par_source[NOMS_SOURCES[src]] = {
            "fiches": nb(bronze.filter(pl.col("source") == src).height),
            "points_faibles": [f"{r['dimension']} - {r['champ']} : {pct(r['taux'])} %" for r in faibles.iter_rows(named=True)],
            "corrections_appliquees": {r["op"]: nb(r["n"]) for r in j.filter(pl.col("type") == "correction").iter_rows(named=True)},
            "anomalies_signalees": {r["op"]: nb(r["n"]) for r in j.filter(pl.col("type") == "anomalie").iter_rows(named=True)},
        }
    totaux = journal.group_by("type", "op").agg(pl.col("n").sum()).sort("n", descending=True)
    decisions = dict(paires.iter_rows())
    inactifs = gold.filter((pl.lit(aujourd_hui) - pl.col("date_derniere_activite")).dt.total_days()
                           > FRAICHEUR_ANS * 365.25).height
    multi = gold.filter(pl.col("n_fiches") > 1)
    return {
        "date": aujourd_hui.strftime("%d/%m/%Y"),
        "nombre_de_fiches_avant_traitement": nb(bronze.height),
        "nombre_de_clients_distincts_apres_traitement": nb(gold.height),
        "fiches_en_double_reunies": nb(bronze.height - gold.height),
        "part_de_fiches_en_double_AVANT_traitement": pct(1 - gold.height / bronze.height),
        "clients_qui_avaient_plusieurs_fiches_et_sont_desormais_reunis": nb(multi.height),
        "clients_reunis_qui_etaient_dans_les_3_systemes": nb(gold.filter(pl.col("sources").list.len() == 3).height),
        "score_qualite_avant_traitement": pct(s_avant), "score_qualite_apres_traitement": pct(s_apres),
        "dimensions": {d: {"avant": pct(d_avant[d]), "apres": pct(d_apres[d])} for d in d_avant},
        "sources": par_source,
        "corrections_appliquees_toutes_sources": {op: nb(n) for t, op, n in totaux.iter_rows() if t == "correction"},
        "anomalies_signalees_toutes_sources": {op: nb(n) for t, op, n in totaux.iter_rows() if t == "anomalie"},
        "cas_douteux_non_fusionnes_en_attente": nb(decisions.get("ambigu", 0)),
        "cas_douteux_deja_examines_par_ia": nb(n_ia),
        "clients_avec_conflit_a_verifier_manuellement": nb(gold.filter(pl.col("conflits").list.len() > 0).height),
        "clients_sans_activite_depuis_plus_de_3_ans": nb(inactifs),
        "part_des_clients_sans_activite_depuis_plus_de_3_ans": pct(inactifs / gold.height),
        "seuil_inactivite_ans": FRAICHEUR_ANS,
    }


PROMPT = """Tu es consultant en qualite des donnees. Redige en francais un rapport d'audit du fichier
clients pour une RESPONSABLE MARKETING (pas une technicienne) : clair, concret, oriente action.

Regles absolues :
- N'utilise QUE les chiffres fournis, recopies EXACTEMENT tels quels (meme format : « 79,1 », « 10 407 »).
  N'en calcule aucun autre, n'arrondis pas, n'invente rien.
- N'attribue un chiffre a une source (CRM, boutique en ligne, service client) QUE s'il figure sous
  cette source dans les faits. Un chiffre "toutes_sources" concerne l'ensemble du fichier : ne le
  rattache jamais a une seule source.
- Les "corrections_appliquees" sont DEJA faites ; les "anomalies_signalees" sont des problemes detectes
  mais non corriges automatiquement.
- Pas de jargon technique (pas de « Jaro-Winkler », « blocage », « silver »...).

Structure (Markdown, titres ##) :
## Synthese  (3 a 4 phrases : l'essentiel a retenir, avec le score avant -> apres)
## Etat initial du fichier  (les problemes principaux et la source d'ou ils viennent)
## Ce qui a ete corrige
## Doublons : ce que le traitement a permis de reunifier
## Points de vigilance  (ce qui reste a faire : cas a verifier, clients inactifs...)
## Recommandations  (4 a 6 actions concretes, chacune rattachee a une source : ex. rendre un champ
   obligatoire dans un formulaire, controler un format a la saisie, campagne de requalification...)

Reponds uniquement avec le rapport en Markdown, sans phrase d'introduction."""


def nombres_autorises(faits: dict) -> set[str]:
    """Tous les nombres presents dans les faits, sous une forme normalisee (sans espaces)."""
    texte = json.dumps(faits, ensure_ascii=False)
    return {m.replace(" ", "") for m in re.findall(r"\d[\d ]*(?:,\d+)?", texte)}


ESPACES = {"\u00a0": " ", "\u202f": " ", "\u2009": " "}  # insecable, insecable fine, fine


def verifier_chiffres(rapport: str, faits: dict) -> list[str]:
    """Renvoie les nombres du rapport qui ne figurent PAS dans les faits (hallucinations probables).

    Limite connue : on verifie les NOMBRES, pas leur SENS. Un chiffre exact attribue a la mauvaise
    source passe ce controle : d'ou les faits detailles par source, et une relecture humaine."""
    for k, v in ESPACES.items():  # l'IA ecrit parfois « 18 170 » avec une espace insecable fine
        rapport = rapport.replace(k, v)
    autorises = nombres_autorises(faits)
    trouves = re.findall(r"(?<![\w-])\d{1,3}(?:[  ]\d{3})+(?:,\d+)?|(?<![\w-])\d+(?:,\d+)?", rapport)
    suspects = []
    for n in trouves:
        cle = n.replace(" ", "").replace(" ", "")
        if cle in autorises or cle == "100" or (cle.isdigit() and int(cle) <= 10):  # « / 100 », numerotation
            continue
        suspects.append(n)
    return sorted(set(suspects))


def rapport_modele(f: dict) -> str:
    """Version sans IA : un modele de texte fixe, rempli avec les faits."""
    lignes = [
        "## Synthese",
        f"Le fichier contenait {f['nombre_de_fiches_avant_traitement']} fiches pour {f['nombre_de_clients_distincts_apres_traitement']} clients distincts identifies : "
        f"{f['part_de_fiches_en_double_AVANT_traitement']} % des fiches etaient redondantes. Apres traitement, le score de qualite "
        f"passe de {f['score_qualite_avant_traitement']} a {f['score_qualite_apres_traitement']} / 100.",
        "", "## Etat initial du fichier",
    ]
    for src, d in f["sources"].items():
        faibles = ", ".join(d["points_faibles"][:4]) or "aucun point faible majeur"
        lignes.append(f"- **{src}** ({d['fiches']} fiches) : {faibles}")
    lignes += ["", "## Ce qui a ete corrige"]
    lignes += [f"- {op} : {n}" for op, n in f["corrections_appliquees_toutes_sources"].items()]
    lignes += ["", "Problemes signales sans correction automatique :"]
    lignes += [f"- {op} : {n}" for op, n in f["anomalies_signalees_toutes_sources"].items()]
    lignes += [
        "", "## Doublons : ce que le traitement a permis de reunifier",
        f"- {f['clients_qui_avaient_plusieurs_fiches_et_sont_desormais_reunis']} clients avaient plusieurs fiches, dont "
        f"{f['clients_reunis_qui_etaient_dans_les_3_systemes']} presents dans les 3 systemes.",
        f"- {f['fiches_en_double_reunies']} fiches redondantes ont ete reunies.",
        "", "## Points de vigilance",
        f"- {f['cas_douteux_non_fusionnes_en_attente']} cas ambigus restent a examiner ({f['cas_douteux_deja_examines_par_ia']} deja analyses par l'IA).",
        f"- {f['clients_avec_conflit_a_verifier_manuellement']} clients presentent un conflit a verifier.",
        f"- {f['clients_sans_activite_depuis_plus_de_3_ans']} clients ({f['part_des_clients_sans_activite_depuis_plus_de_3_ans']} %) sont inactifs depuis plus de "
        f"{f['seuil_inactivite_ans']} ans.",
        "", "## Recommandations",
        "- Controler le format de l'e-mail et du telephone des la saisie, dans chaque formulaire.",
        "- Demander au service client de renseigner systematiquement le prenom.",
        "- Lancer une campagne de requalification des clients inactifs.",
        "- Faire examiner en priorite les clients signales en conflit.",
    ]
    return "\n".join(lignes)


def rediger(faits: dict, appeler=None) -> tuple[str, str, list[str]]:
    """Renvoie (rapport, auteur, chiffres_suspects). `appeler(messages) -> str` permet de tester sans API."""
    if appeler is None:
        import os
        if not os.environ.get("GROQ_API_KEY"):
            return rapport_modele(faits), "modele fixe (pas de cle API)", []
        from groq import Groq

        # Un seul appel : on prend le grand modele, meilleur redacteur (impact negligeable sur le quota)
        MODELE = os.environ.get("QDC_MODELE_RAPPORT", "openai/gpt-oss-120b")

        def appeler(messages):
            r = Groq().chat.completions.create(model=MODELE, messages=messages, temperature=0.2, max_tokens=3000)
            return r.choices[0].message.content or ""
    try:
        texte = appeler([{"role": "system", "content": PROMPT},
                         {"role": "user", "content": json.dumps(faits, ensure_ascii=False, indent=1)}]).strip()
    except Exception as e:
        print(f"  appel IA impossible ({type(e).__name__}) : rapport genere par le modele fixe")
        return rapport_modele(faits), "modele fixe (IA indisponible)", []
    if not texte.startswith("##"):
        texte = texte[texte.find("##"):] if "##" in texte else texte
    return texte, "IA", verifier_chiffres(texte, faits)


def assembler(faits: dict, corps: str, auteur: str, suspects: list[str]) -> str:
    entete = [f"# Audit qualite du fichier clients - {faits['date']}", "",
              f"*Redige par : {auteur}. Chiffres calcules par le pipeline de donnees. "
              "Document a relire avant diffusion.*", ""]
    if suspects:
        entete += [f"> ⚠️ Chiffres non retrouves dans les donnees : {', '.join(suspects)}. "
                   "A verifier avant diffusion.", ""]
    elif auteur == "IA":
        entete += ["> ✅ Tous les chiffres du rapport ont ete verifies dans les donnees.", ""]
    return "\n".join(entete) + corps + "\n"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    base = Path("data/qdc.duckdb")
    faits = collecter_faits(base)
    corps, auteur, suspects = rediger(faits)
    rapport = assembler(faits, corps, auteur, suspects)
    sortie = Path("data/rapport_audit.md")
    sortie.write_text(rapport, encoding="utf-8")
    print(rapport)
    print(f"\n  -> rapport enregistre dans {sortie}  (redige par : {auteur} ; chiffres suspects : {len(suspects)})")
