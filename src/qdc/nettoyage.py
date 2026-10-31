"""Nettoyage : de la couche bronze (brute) a la couche silver (propre et typee).

Regle d'or : on CORRIGE ce qui est sur, on SIGNALE le reste, on ne DEVINE jamais.
- "06 12 34 56 78" -> "+33612345678"     : sur, on corrige
- "matthieu.thomashotmail.fr"            : on ne sait pas ou mettre le @ -> on signale, on vide
- "35000 REIMS"                          : qui a raison, le CP ou la ville ? -> on signale seulement

Chaque ligne garde une trace de ce qu'on lui a fait :
- `corrections` : la liste des corrections appliquees
- `anomalies`   : la liste des problemes detectes mais non corriges
- les valeurs brutes d'origine restent dans la couche bronze (lien par record_id)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from qdc import regles
from qdc.referentiel import PRENOMS, PRENOMS_FEMININS, PRENOMS_MASCULINS, VILLE_PAR_CP

CIVILITES = {"M.": "M.", "M": "M.", "MR": "M.", "MONSIEUR": "M.",
             "MME": "Mme", "MADAME": "Mme", "MLLE": "Mme"}


def _est_prenom(e: pl.Expr) -> pl.Expr:
    """Un mot est un prenom s'il est dans le referentiel, ou si c'est une initiale ("J.")."""
    cle = regles.norm_ville(e)  # majuscules sans accents, meme cle que le referentiel
    return cle.is_in(list(PRENOMS)) | cle.str.contains(r"^[A-Z]\.$")


def separer_nom_complet(df: pl.DataFrame) -> pl.DataFrame:
    """Decoupe "Nathalie TOUSSAINT" / "TOUSSAINT Nathalie" / "M. Petit" en civilite, prenom, nom.

    L'ordre des mots varie selon la saisie : on regarde quel mot est un prenom connu.
    Si on ne peut pas trancher, on suppose l'ordre le plus frequent (prenom puis nom)
    et on le SIGNALE.
    """
    a_decouper = pl.col("nom").is_null() & pl.col("nom_complet").is_not_null()
    mots = regles.norm_texte(pl.col("nom_complet")).str.split(" ")

    # Civilite en tete ("M. Petit", "Mme Durand")
    premier = mots.list.first().str.to_uppercase()
    a_civ = premier.is_in(list(CIVILITES))
    df = df.with_columns(
        _civ=pl.when(a_decouper & a_civ).then(premier),
        _mots=pl.when(a_decouper & a_civ).then(mots.list.slice(1)).when(a_decouper).then(mots),
    )
    debut, fin = pl.col("_mots").list.first(), pl.col("_mots").list.last()
    n_mots = pl.col("_mots").list.len()
    deb_prenom, fin_prenom = _est_prenom(debut), _est_prenom(fin)
    sans_prenom = pl.col("prenom").is_null()

    ordre = (
        pl.when(pl.col("_mots").is_null()).then(None)
        .when(n_mots == 1).then(pl.lit("nom_seul"))
        .when(deb_prenom & ~fin_prenom).then(pl.lit("prenom_nom"))
        .when(fin_prenom & ~deb_prenom).then(pl.lit("nom_prenom"))
        .otherwise(pl.lit("incertain"))  # les deux, ou aucun, ressemblent a un prenom
    )
    df = df.with_columns(_ordre=ordre)
    reste_apres_debut = pl.col("_mots").list.slice(1).list.join(" ")
    reste_avant_fin = pl.col("_mots").list.slice(0, n_mots - 1).list.join(" ")

    return df.with_columns(
        civilite=pl.coalesce("civilite", pl.col("_civ")),
        prenom=pl.when(sans_prenom & pl.col("_ordre").is_in(["prenom_nom", "incertain"])).then(debut)
              .when(sans_prenom & (pl.col("_ordre") == "nom_prenom")).then(fin)
              .otherwise(pl.col("prenom")),
        nom=pl.when(pl.col("_ordre") == "nom_seul").then(pl.col("_mots").list.join(" "))
            .when(pl.col("_ordre") == "nom_prenom").then(reste_avant_fin)
            .when(pl.col("_ordre").is_in(["prenom_nom", "incertain"]) & sans_prenom).then(reste_apres_debut)
            # le prenom est deja connu (service client) : on l'enleve s'il est repete
            .when(pl.col("_ordre").is_not_null()).then(pl.col("_mots").list.join(" "))
            .otherwise(pl.col("nom")),
    )


def nettoyer(bronze: pl.DataFrame, aujourd_hui: date | None = None) -> pl.DataFrame:
    ref = pl.lit(aujourd_hui or date.today())
    df = bronze.with_columns(
        [regles.norm_texte(pl.col(c)).alias(c) for c in
         ("civilite", "prenom", "nom", "nom_complet", "adresse", "ville")]
    )
    df = separer_nom_complet(df)

    # Valeurs normalisees
    email_n = regles.norm_email(pl.col("email"))
    tel_n = regles.norm_telephone(pl.col("telephone"))
    cp_n = regles.norm_cp(pl.col("code_postal"))
    civ_n = pl.col("civilite").str.to_uppercase().str.replace_all(r"\s", "").replace_strict(
        CIVILITES, default=None, return_dtype=pl.String)
    prenom_maj = regles.norm_ville(pl.col("prenom"))
    civ_deduite = (pl.when(prenom_maj.is_in(list(PRENOMS_FEMININS))).then(pl.lit("Mme"))
                   .when(prenom_maj.is_in(list(PRENOMS_MASCULINS))).then(pl.lit("M.")))
    ville_ref = cp_n.replace_strict(VILLE_PAR_CP, default=None, return_dtype=pl.String)
    st_email, st_tel = regles.statut_email("email"), regles.statut_telephone("telephone")
    st_dn = regles.statut_date_naissance("date_naissance", ref)

    df = df.with_columns(
        _st_email=st_email, _st_tel=st_tel, _st_dn=st_dn,
        _coherent=regles.cp_coherent("code_postal", "ville"),
        _ville_ref=ville_ref, _civ_n=civ_n, _civ_deduite=civ_deduite,
        _ordre_incertain=pl.col("_ordre") == "incertain",
    )

    # Journal : ce qu'on a corrige / ce qu'on a seulement signale
    def si(cond: pl.Expr, texte: str) -> pl.Expr:
        return pl.when(cond).then(pl.lit(texte))

    corrections = pl.concat_list([
        si(pl.col("_st_email") == "corrigeable", "email: format"),
        si(pl.col("_st_tel") == "corrigeable", "telephone: format E.164"),
        si(pl.col("nom_complet").is_not_null(), "nom: decoupe du nom complet"),
        si(pl.col("_civ_n").is_null() & pl.col("_civ_deduite").is_not_null(), "civilite: deduite du prenom"),
        si(pl.col("ville").is_null() & pl.col("_ville_ref").is_not_null(), "ville: deduite du code postal"),
    ]).list.drop_nulls()
    anomalies = pl.concat_list([
        si(pl.col("_st_email") == "invalide", "email invalide (supprime)"),
        si(pl.col("_st_tel") == "invalide", "telephone invalide (supprime)"),
        si(pl.col("_st_dn") == "invalide", "date de naissance aberrante (supprimee)"),
        si(pl.col("_coherent") == False, "code postal incoherent avec la ville"),  # noqa: E712
        si(pl.col("_ordre_incertain"), "ordre prenom/nom incertain"),
    ]).list.drop_nulls()

    silver = df.with_columns(
        civilite=pl.coalesce("_civ_n", "_civ_deduite"),
        prenom=pl.col("prenom").str.to_titlecase(),
        nom=pl.col("nom").str.to_titlecase(),
        email=pl.when(pl.col("_st_email") != "invalide").then(email_n),
        telephone=pl.when(pl.col("_st_tel") != "invalide").then(tel_n),
        code_postal=cp_n,
        ville=pl.coalesce(pl.col("ville"), pl.col("_ville_ref")).str.to_titlecase(),
        date_naissance=pl.when(pl.col("_st_dn") == "valide").then(regles.en_date(pl.col("date_naissance"))),
        date_creation=regles.en_date(pl.col("date_creation")),
        date_dernier_contact=regles.en_date(pl.col("date_dernier_contact")),
        corrections=corrections,
        anomalies=anomalies,
    )
    return silver.select(
        "record_id", "source", "civilite", "prenom", "nom", "email", "telephone", "adresse",
        "code_postal", "ville", "date_naissance", "date_creation", "date_dernier_contact",
        "corrections", "anomalies",
    )


def journal(silver: pl.DataFrame) -> pl.DataFrame:
    """Combien de corrections / d'anomalies de chaque type ?"""
    morceaux = []
    for type_, col in (("correction", "corrections"), ("anomalie", "anomalies")):
        morceaux.append(silver.select(pl.col(col).explode().alias("operation")).drop_nulls()
                        .group_by("operation").len("n").with_columns(type=pl.lit(type_)))
    return pl.concat(morceaux).select("type", "operation", "n").sort(["type", "n"], descending=[False, True])


def pour_diagnostic(silver: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, set[str]]]:
    """Le diagnostic travaille sur du texte : on reconvertit les dates.
    Apres decoupe, les sources qui avaient un "nom complet" ont desormais prenom + nom."""
    from qdc.diagnostic import APPLICABLE_BRONZE
    applicable = {
        s: (champs - {"nom_complet"}) | ({"prenom", "nom"} if "nom_complet" in champs else set())
        for s, champs in APPLICABLE_BRONZE.items()
    }
    return silver.with_columns(pl.col(pl.Date).cast(pl.String)), applicable


if __name__ == "__main__":
    from qdc import diagnostic

    base = Path("data/qdc.duckdb")
    with duckdb.connect(str(base)) as con:
        bronze = con.sql("SELECT * FROM bronze_clients").pl()
    silver = nettoyer(bronze)
    with duckdb.connect(str(base)) as con:
        con.register("s", silver.to_arrow())
        con.execute("CREATE OR REPLACE TABLE silver_clients AS SELECT * FROM s")

    print("\n  JOURNAL DU NETTOYAGE")
    for r in journal(silver).iter_rows(named=True):
        print(f"    {r['type']:<11} {r['operation']:<42} {r['n']:>6}")

    avant = diagnostic.diagnostiquer(bronze)
    df_s, applicable = pour_diagnostic(silver)
    apres = diagnostic.diagnostiquer(df_s, applicable=applicable)
    diagnostic.sauvegarder(apres, base, "apres_nettoyage")
    print(f"\n  SCORE : {avant.score_global} -> {apres.score_global} / 100\n")
    a = dict(avant.resume.select("dimension", "taux").iter_rows())
    for dim, taux in apres.resume.select("dimension", "taux").iter_rows():
        print(f"    {dim:<11} {100 * a[dim]:5.1f} %  ->  {100 * taux:5.1f} %")
    print("\n  CONFORMITE AU FORMAT DE REFERENCE (ce que le nettoyage ameliore vraiment)")
    for champ in ("email", "telephone", "code_postal"):
        c = [d.details.filter((pl.col("dimension") == "Validité") & (pl.col("champ") == champ))
             .select(pl.col("n_conforme").sum() / pl.col("n_concernes").sum()).item() for d in (avant, apres)]
        print(f"    {champ:<11} {100 * c[0]:5.1f} %  ->  {100 * c[1]:5.1f} %")
