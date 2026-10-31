"""Phase 5 : la fiche client unique (couche GOLD, ou "referentiel client unique").

Pour chaque groupe d'enregistrements qui designent le meme client, on fabrique UNE fiche en
choisissant la meilleure valeur de chaque champ : ce sont les "regles de survie" (survivorship).

Trois principes :
1. Les champs qui vont ensemble restent ensemble : on ne colle pas la rue d'une fiche avec le
   code postal d'une autre. L'adresse complete vient d'UNE seule fiche.
2. Lignage : pour chaque champ retenu, on garde l'identifiant de la fiche d'origine.
   « D'ou vient cette adresse ? » -> on sait repondre.
3. Les conflits sont signales, pas caches (ex : deux dates de naissance dans un meme groupe).

Les liens viennent de deux sources : les doublons surs des regles (phase 4a) et, s'il y en a
deja, les decisions de l'IA jugees fiables (phase 4b, lues dans le cache : aucun appel API ici).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import polars as pl

from qdc import regles

# Fiabilite des sources : le client saisit lui-meme ses informations sur la boutique ;
# le service client les saisit vite, au telephone.
PRIORITE_SOURCE = {"boutique": 0, "crm": 1, "support": 2}


def liens_ia(base: Path, silver: pl.DataFrame, paires: pl.DataFrame, confiance_min: float) -> pl.DataFrame:
    """Les paires ambigues que l'IA a jugees « meme personne » avec assez de confiance."""
    from qdc import ia

    with duckdb.connect(str(base)) as con:
        if "cache_ia" not in {t[0] for t in con.execute("SHOW TABLES").fetchall()}:
            return pl.DataFrame(schema={"id_a": pl.String, "id_b": pl.String})
    ambigues = paires.filter(pl.col("decision") == "ambigu").select("id_a", "id_b")
    questions = ia.preparer_lot(ambigues, silver)
    index = pl.DataFrame({"id": [q["id"] for q in questions], "cle": [q["_cle"] for q in questions]})
    oui = index.join(ia.lire_cache(base), on="cle").filter(
        pl.col("meme_personne") & (pl.col("confiance") >= confiance_min))
    return oui.select(pl.col("id").str.split("|").list.to_struct(fields=["id_a", "id_b"])).unnest("id")


def _prioriser(df: pl.DataFrame) -> pl.DataFrame:
    """Colonnes d'aide au choix : recence, fiabilite de la source, frequence des valeurs."""
    k = regles.norm_ville
    return df.with_columns(
        recence=pl.max_horizontal("date_creation", "date_dernier_contact"),
        prio_source=pl.col("source").replace_strict(PRIORITE_SOURCE, default=9),
        est_initiale=pl.col("prenom").str.contains(r"^[A-Z]\.$").fill_null(True),
        est_mobile=pl.col("telephone").str.contains(r"^\+33[67]").fill_null(False),
        nom_cle=k(pl.col("nom")),
        prenom_cle=k(pl.col("prenom")),
    ).with_columns(
        freq_prenom=pl.len().over("client", "prenom_cle"),
        freq_nom=pl.len().over("client", "nom_cle"),
        freq_civ=pl.len().over("client", "civilite"),
        freq_dn=pl.len().over("client", "date_naissance"),
    )


def _meilleur(df: pl.DataFrame, colonnes: list[str], tri: list[pl.Expr], desc: list[bool],
              requis: str | None = None) -> pl.DataFrame:
    """Pour chaque client, la fiche la mieux classee (qui a la valeur `requis`) fournit `colonnes`."""
    requis = requis or colonnes[0]
    meilleurs = (df.filter(pl.col(requis).is_not_null())
                 .sort(["client", *tri], descending=[False, *desc], nulls_last=True)
                 .group_by("client", maintain_order=True).first())
    return meilleurs.select("client", *colonnes, pl.col("record_id").alias(f"origine_{requis}"))


def fusionner(silver: pl.DataFrame, groupes: pl.DataFrame) -> pl.DataFrame:
    df = _prioriser(silver.join(groupes, on="record_id"))
    r, p = pl.col("recence"), pl.col("prio_source")

    gold = df.group_by("client").agg(
        n_fiches=pl.len(),
        sources=pl.col("source").unique().sort(),
        record_ids=pl.col("record_id").sort(),
        emails=pl.col("email").drop_nulls().unique().sort(),
        telephones=pl.col("telephone").drop_nulls().unique().sort(),
        date_premier_contact=pl.min_horizontal("date_creation", "date_dernier_contact").min(),
        date_derniere_activite=r.max(),
        _n_dn=pl.col("date_naissance").drop_nulls().n_unique(),
        _n_civ=pl.col("civilite").drop_nulls().n_unique(),
        variantes_nom=pl.col("nom").drop_nulls().unique().sort(),
    )
    choix = [
        # prenom : pas une initiale, puis la valeur MAJORITAIRE, puis la source la plus fiable.
        # (Piege evite : « le plus long » choisissait les fautes qui doublent une lettre, « Luccie ».)
        _meilleur(df, ["prenom"], [pl.col("est_initiale"), pl.col("freq_prenom"), p, r], [False, True, False, True]),
        # nom, civilite, date de naissance : la valeur majoritaire, puis la source la plus fiable
        _meilleur(df, ["nom"], [pl.col("freq_nom"), p, r], [True, False, True]),
        _meilleur(df, ["civilite"], [pl.col("freq_civ"), p, r], [True, False, True]),
        _meilleur(df, ["date_naissance"], [pl.col("freq_dn"), p, r], [True, False, True]),
        # e-mail : le plus recent (les autres restent dans la liste `emails`)
        _meilleur(df, ["email"], [r, p], [True, False]),
        # telephone : un mobile de preference, puis le plus recent
        _meilleur(df, ["telephone"], [pl.col("est_mobile"), r], [True, True]),
        # adresse : UN bloc coherent, pris dans la fiche la plus recente qui a une adresse
        _meilleur(df.with_columns(_a_adresse=pl.col("adresse").is_not_null()),
                  ["code_postal", "ville", "adresse"], [pl.col("_a_adresse"), r, p], [True, True, False],
                  requis="code_postal"),
    ]
    for c in choix:
        gold = gold.join(c, on="client", how="left")

    # Mesure sur nos donnees : « dates de naissance differentes » revele presque toujours une
    # FAUSSE FUSION (deux personnes melangees) -> a verifier en priorite. Les variantes
    # d'orthographe du nom, elles, sont des fautes de frappe : une information, pas un conflit.
    conflits = pl.concat_list([
        pl.when(pl.col("_n_dn") > 1).then(pl.lit("dates de naissance differentes")),
        pl.when(pl.col("_n_civ") > 1).then(pl.lit("civilites differentes")),
    ]).list.drop_nulls()

    def identifiant(rid: str) -> str:  # stable tant que les fiches d'origine ne changent pas
        return "CLI-" + hashlib.sha1(rid.encode()).hexdigest()[:10].upper()

    return gold.with_columns(
        client_id=pl.col("record_ids").list.first().map_elements(identifiant, return_dtype=pl.String),
        conflits=conflits,
    ).select(
        "client_id", "civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal", "ville",
        "date_naissance", "date_premier_contact", "date_derniere_activite", "n_fiches", "sources",
        "emails", "telephones", "variantes_nom", "conflits", "record_ids",
        pl.struct(pl.col("^origine_.*$")).alias("lignage"),
    ).sort("client_id")


def pour_diagnostic(gold: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, set[str]]]:
    """Met le referentiel au format attendu par le diagnostic, pour comparer avant / apres."""
    df = gold.select(
        pl.col("client_id").alias("record_id"), pl.lit("referentiel").alias("source"),
        "civilite", "prenom", "nom", pl.lit(None, dtype=pl.String).alias("nom_complet"), "email", "telephone",
        "adresse", "code_postal", "ville", pl.col("date_naissance").cast(pl.String),
        pl.col("date_premier_contact").cast(pl.String).alias("date_creation"),
        pl.col("date_derniere_activite").cast(pl.String).alias("date_dernier_contact"),
    )
    champs = {"civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal", "ville",
              "date_naissance", "date_creation", "date_dernier_contact"}
    return df, {"referentiel": champs}


if __name__ == "__main__":
    import argparse

    from qdc import diagnostic
    from qdc.dedoublonnage import evaluer, regrouper
    from qdc.ia import CONFIANCE_FUSION

    parser = argparse.ArgumentParser(description="Construit le referentiel client unique.")
    parser.add_argument("--sans-ia", action="store_true", help="ignorer les decisions de l'IA")
    args = parser.parse_args()

    base = Path("data/qdc.duckdb")
    with duckdb.connect(str(base)) as con:
        bronze = con.sql("SELECT * FROM bronze_clients").pl()
        silver = con.sql("SELECT * FROM silver_clients").pl()
        paires = con.sql("SELECT * FROM paires").pl()

    liens = paires.filter(pl.col("decision") == "doublon").select("id_a", "id_b")
    ia_ok = pl.DataFrame(schema=liens.schema) if args.sans_ia else liens_ia(base, silver, paires, CONFIANCE_FUSION)
    groupes = regrouper(silver["record_id"], pl.concat([liens, ia_ok]))
    gold = fusionner(silver, groupes)

    with duckdb.connect(str(base)) as con:
        con.register("g", gold.to_arrow())
        con.execute("CREATE OR REPLACE TABLE gold_clients AS SELECT * FROM g")
        con.register("m", groupes.join(gold.select("client_id", pl.col("record_ids")).explode("record_ids"),
                                       left_on="record_id", right_on="record_ids").select("record_id", "client_id").to_arrow())
        con.execute("CREATE OR REPLACE TABLE correspondance_clients AS SELECT * FROM m")

    print(f"\n  REFERENTIEL CLIENT UNIQUE : {silver.height:,} fiches -> {gold.height:,} clients".replace(",", " "))
    print(f"    liens : {liens.height:,} des regles + {ia_ok.height:,} de l'IA".replace(",", " "))
    print(f"    clients avec plusieurs fiches : {gold.filter(pl.col('n_fiches') > 1).height:,}".replace(",", " "))
    print(f"    clients avec un conflit signale : {gold.filter(pl.col('conflits').list.len() > 0).height:,}".replace(",", " "))

    ev = evaluer(groupes, pl.read_csv("data/verite/verite.csv"))
    print(f"    contre la verite : {ev.clients_reels:,} vrais clients, precision {100 * ev.precision:.1f} %, "
          f"rappel {100 * ev.rappel:.1f} %".replace(",", " "))
    print(f"    unicite REELLE : {100 * ev.clients_reels / gold.height:.1f} %  (le diagnostic ne voit que les "
          f"doublons evidents ; il reste des doublons parmi les cas ambigus)")

    avant = diagnostic.diagnostiquer(bronze)
    df_g, applicable = pour_diagnostic(gold)
    apres = diagnostic.diagnostiquer(df_g, applicable=applicable)
    diagnostic.sauvegarder(apres, base, "referentiel")
    print(f"\n  SCORE DE QUALITE : {avant.score_global} -> {apres.score_global} / 100\n")
    a = dict(avant.resume.select("dimension", "taux").iter_rows())
    for dim, taux in apres.resume.select("dimension", "taux").iter_rows():
        print(f"    {dim:<11} {100 * a[dim]:5.1f} %  ->  {100 * taux:5.1f} %")

    print("\n  EXEMPLE DE FICHE UNIQUE (client avec le plus de fiches)")
    ex = gold.sort("n_fiches", descending=True).row(0, named=True)
    for c in ("client_id", "civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal",
              "ville", "date_naissance", "sources", "emails", "conflits", "record_ids", "lignage"):
        print(f"    {c:<15} {ex[c]}")
