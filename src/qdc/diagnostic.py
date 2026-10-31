"""Diagnostic : mesurer la qualite des donnees AVANT de les corriger.

On mesure 5 dimensions classiques de la qualite des donnees :
- Completude  : la donnee est-elle renseignee ?
- Validite    : respecte-t-elle le format attendu ?
- Unicite     : chaque client n'apparait-il qu'une fois ?
- Coherence   : les champs sont-ils coherents entre eux (CP <-> ville) ?
- Fraicheur   : la donnee a-t-elle ete mise a jour recemment ?
(L'exactitude - "la donnee est-elle vraie ?" - demande une source de verite externe :
on la mesurera a la phase 4 grace au fichier verite.)

Regle importante : un champ qu'une source ne collecte PAS n'est pas "manquant",
il est "non applicable". Sinon on penaliserait le service client parce qu'il ne
demande pas la date de naissance, ce qui fausserait tout le diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from qdc import regles
from qdc.ingestion import SOURCES

CHAMPS_COMPLETUDE = ["civilite", "prenom", "nom", "nom_complet", "email", "telephone",
                     "adresse", "code_postal", "ville", "date_naissance"]
CHAMPS_VALIDITE = {
    "email": regles.statut_email,
    "telephone": regles.statut_telephone,
    "code_postal": regles.statut_cp,
}
FRAICHEUR_ANS = 3

# Quels champs chaque source collecte-t-elle ?
APPLICABLE = {s.nom: set(s.correspondance.values()) for s in SOURCES}


@dataclass
class Diagnostic:
    details: pl.DataFrame   # une ligne par (dimension, champ, source)
    resume: pl.DataFrame    # une ligne par dimension
    score_global: float     # 0 a 100
    exemples: dict[str, list[str]]
    n_lignes: int


def _ligne(dimension, champ, source, n, ok, conforme=None, agrege=True):
    return {"dimension": dimension, "champ": champ, "source": source,
            "n_concernes": n, "n_ok": ok, "taux": (ok / n) if n else None,
            "n_conforme": conforme, "agrege": agrege}


def groupes_evidents(df: pl.DataFrame) -> pl.Series:
    """Regroupe les enregistrements qui partagent un e-mail OU un telephone (normalises).

    Si A et B ont le meme e-mail, et B et C le meme telephone, alors A, B et C forment
    UN seul client : c'est un probleme de "composantes connexes" dans un graphe.
    On le resout par propagation d'etiquettes : chaque ligne prend le plus petit
    identifiant de son groupe, et on recommence jusqu'a ce que plus rien ne bouge.
    Tout est vectorise (pas de boucle ligne par ligne) : ca tient sur des millions de lignes.
    """
    g = df.select(
        regles.norm_email(pl.col("email")).alias("email_n"),
        regles.norm_telephone(pl.col("telephone")).alias("tel_n"),
    ).with_row_index("groupe")
    for _ in range(50):
        avant = g["groupe"]
        for cle in ("email_n", "tel_n"):
            g = g.with_columns(
                pl.when(pl.col(cle).is_not_null())
                .then(pl.min_horizontal("groupe", pl.col("groupe").min().over(cle)))
                .otherwise(pl.col("groupe")).alias("groupe")
            )
        if g["groupe"].equals(avant):
            break
    return g["groupe"]


def diagnostiquer(df: pl.DataFrame, aujourd_hui: date | None = None) -> Diagnostic:
    aujourd_hui = aujourd_hui or date.today()
    ref = pl.lit(aujourd_hui)
    lignes: list[dict] = []
    exemples: dict[str, list[str]] = {}

    # 1. Completude et validite, source par source
    for source, part in df.group_by("source", maintain_order=True):
        source = source[0]
        n = part.height
        for champ in CHAMPS_COMPLETUDE:
            if champ not in APPLICABLE[source]:
                continue
            ok = part.select((~regles.vide(pl.col(champ))).sum()).item()
            lignes.append(_ligne("Complétude", champ, source, n, ok))

        statuts = {c: f(c) for c, f in CHAMPS_VALIDITE.items() if c in APPLICABLE[source]}
        if "date_naissance" in APPLICABLE[source]:
            statuts["date_naissance"] = regles.statut_date_naissance("date_naissance", ref)
        st = part.select([e.alias(c) for c, e in statuts.items()] + [pl.col(c).alias(f"_{c}") for c in statuts])
        for champ in statuts:
            renseignes = st.filter(pl.col(champ) != "manquante")
            # "Exploitable" = valide OU corrigeable sans risque : c'est ce qui compte pour le score.
            # "Conforme" = deja au format de reference : on le garde pour savoir combien corriger.
            exploitable = renseignes.filter(pl.col(champ) != "invalide").height
            conforme = renseignes.filter(pl.col(champ) == "valide").height
            lignes.append(_ligne("Validité", champ, source, renseignes.height, exploitable, conforme))
            for etat in ("corrigeable", "invalide"):
                vals = renseignes.filter(pl.col(champ) == etat)[f"_{champ}"].head(4).to_list()
                if vals:
                    exemples.setdefault(f"{champ} {etat}", []).extend(vals)

        # 2. Coherence CP <-> ville (seulement si la source a les deux champs)
        if {"code_postal", "ville"} <= APPLICABLE[source]:
            coh = part.select(regles.cp_coherent("code_postal", "ville").alias("c"),
                              pl.col("code_postal"), pl.col("ville")).filter(pl.col("c").is_not_null())
            lignes.append(_ligne("Cohérence", "code_postal ↔ ville", source, coh.height, int(coh["c"].sum())))
            ko = coh.filter(~pl.col("c")).head(4)
            exemples.setdefault("CP incohérent avec la ville", []).extend(
                f"{r['code_postal']} {r['ville']}" for r in ko.iter_rows(named=True))

        # 3. Fraicheur : derniere activite connue
        derniere = pl.max_horizontal(regles.en_date(pl.col("date_creation")),
                                     regles.en_date(pl.col("date_dernier_contact")))
        fr = part.select(derniere.alias("d")).filter(pl.col("d").is_not_null())
        recent = fr.filter((ref - pl.col("d")).dt.total_days() <= FRAICHEUR_ANS * 365.25).height
        lignes.append(_ligne("Fraîcheur", f"activité < {FRAICHEUR_ANS} ans", source, fr.height, recent))

    # 4. Unicite : doublons EVIDENTS (meme e-mail ou meme telephone apres normalisation).
    #    Estimation basse : les doublons approximatifs (fautes de frappe, e-mail different...)
    #    seront detectes a la phase 4. Unicite = nombre de clients distincts / nombre de lignes.
    grp = df.select("source").with_columns(groupes_evidents(df))
    lignes.append(_ligne("Unicité", "clients distincts / lignes", "toutes", grp.height, grp["groupe"].n_unique()))
    for source, part in grp.group_by("source", maintain_order=True):
        lignes.append(_ligne("Unicité", "doublons internes", source[0], part.height,
                             part["groupe"].n_unique(), agrege=False))

    details = pl.DataFrame(lignes)
    resume = (
        details.filter(pl.col("agrege")).group_by("dimension", maintain_order=True)
        .agg(n_concernes=pl.col("n_concernes").sum(), n_ok=pl.col("n_ok").sum())
        .with_columns(taux=pl.col("n_ok") / pl.col("n_concernes"))
    )
    score = round(100 * resume["taux"].mean(), 1)
    return Diagnostic(details, resume, score, exemples, df.height)


def sauvegarder(diag: Diagnostic, base: Path, etape: str = "avant") -> None:
    with duckdb.connect(str(base)) as con:
        con.register("d", diag.details.with_columns(pl.lit(etape).alias("etape")).to_arrow())
        con.execute("CREATE OR REPLACE TABLE diagnostic_" + etape + " AS SELECT * FROM d")


def afficher(diag: Diagnostic) -> None:
    print(f"\n  SCORE DE QUALITE GLOBAL : {diag.score_global} / 100   ({diag.n_lignes:,} lignes)\n".replace(",", " "))
    for r in diag.resume.iter_rows(named=True):
        barre = "█" * round(r["taux"] * 30)
        print(f"  {r['dimension']:<11} {barre:<30} {100 * r['taux']:5.1f} %")
    print("\n  Pires champs :")
    pires = diag.details.filter(pl.col("taux") < 0.95).sort("taux").head(8)
    for r in pires.iter_rows(named=True):
        print(f"    {r['dimension']:<11} {r['champ']:<30} {r['source']:<9} {100 * r['taux']:5.1f} %")
    print("\n  Exemples de problemes :")
    for k, v in diag.exemples.items():
        print(f"    {k:<28} {v[:3]}")


if __name__ == "__main__":
    base = Path("data/qdc.duckdb")
    with duckdb.connect(str(base), read_only=True) as con:
        bronze = con.sql("SELECT * FROM bronze_clients").pl()
    diag = diagnostiquer(bronze)
    afficher(diag)
    sauvegarder(diag, base)
