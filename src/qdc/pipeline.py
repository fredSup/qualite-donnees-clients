"""Orchestration : enchaine toutes les etapes, de la donnee brute au referentiel.

L'interface (app/) n'appelle QUE ce module : toute la logique reste dans le moteur, deja teste.
C'est la separation interface / logique metier : on pourrait remplacer Streamlit par une API
ou un ordonnanceur (Airflow, Databricks Jobs) sans toucher a une seule regle.

Les decisions humaines (file de revue) sont stockees dans la table `decisions_humaines` et
priment sur tout : un humain qui dit « garder separes » annule l'avis de l'IA.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import polars as pl

from qdc import dedoublonnage, fusion, ingestion, nettoyage  # noqa: F401
from qdc.generation import Config, generer
from qdc.ia import CONFIANCE_FUSION

Etape = Callable[[str], None]


def _ecrire(base: Path, tables: dict[str, pl.DataFrame]) -> None:
    with duckdb.connect(str(base)) as con:
        for nom, df in tables.items():
            con.register("t", df.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {nom} AS SELECT * FROM t")
            con.unregister("t")


def _tables(base: Path) -> set[str]:
    if not base.exists():
        return set()
    with duckdb.connect(str(base)) as con:
        return {t[0] for t in con.execute("SHOW TABLES").fetchall()}


def lire(base: Path, table: str) -> pl.DataFrame:
    with duckdb.connect(str(base)) as con:
        return con.sql(f"SELECT * FROM {table}").pl()


# --- Decisions humaines ------------------------------------------------------------
def _decisions_table(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS decisions_humaines (
        id_a VARCHAR, id_b VARCHAR, decision VARCHAR, decide_le TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (id_a, id_b))""")


def enregistrer_decision(base: Path, id_a: str, id_b: str, decision: str) -> None:
    """decision : 'fusionner' ou 'separer'."""
    assert decision in {"fusionner", "separer"}
    with duckdb.connect(str(base)) as con:
        _decisions_table(con)
        con.execute("INSERT OR REPLACE INTO decisions_humaines (id_a, id_b, decision) VALUES (?, ?, ?)",
                    [id_a, id_b, decision])


def decisions_humaines(base: Path) -> pl.DataFrame:
    with duckdb.connect(str(base)) as con:
        _decisions_table(con)
        return con.sql("SELECT id_a, id_b, decision FROM decisions_humaines").pl()


# --- Etapes ------------------------------------------------------------------------
def generer_demo(dossier: Path, n: int = 10_000, seed: int = 42) -> dict[str, int]:
    return generer(Config(n_personnes=n, seed=seed), dossier)


def construire_referentiel(base: Path, avec_ia: bool = True) -> pl.DataFrame:
    """Fusion finale : doublons surs + avis fiables de l'IA + decisions humaines (qui priment)."""
    silver, paires = lire(base, "silver_clients"), lire(base, "paires")
    liens = paires.filter(pl.col("decision") == "doublon").select("id_a", "id_b")
    if avec_ia:
        liens = pl.concat([liens, fusion.liens_ia(base, silver, paires, CONFIANCE_FUSION)])
    humain = decisions_humaines(base)
    separer = humain.filter(pl.col("decision") == "separer").select("id_a", "id_b")
    liens = pl.concat([liens.join(separer, on=["id_a", "id_b"], how="anti"),
                       humain.filter(pl.col("decision") == "fusionner").select("id_a", "id_b")]).unique()
    groupes = dedoublonnage.regrouper(silver["record_id"], liens)
    extras = lire(base, "colonnes_extra") if "colonnes_extra" in _tables(base) else None
    gold = fusion.fusionner(silver, groupes, extras)
    correspondance = groupes.join(gold.select("client_id", pl.col("record_ids")).explode("record_ids"),
                                  left_on="record_id", right_on="record_ids").select("record_id", "client_id")
    _ecrire(base, {"groupes_clients": groupes, "gold_clients": gold, "correspondance_clients": correspondance})
    return gold


def _traiter(bronze: pl.DataFrame, base: Path, etape: Etape) -> dict:
    etape("Nettoyage et validation")
    silver = nettoyage.nettoyer(bronze)
    _ecrire(base, {"silver_clients": silver})
    etape("Recherche des doublons")
    paires, _ = dedoublonnage.executer(silver)
    _ecrire(base, {"paires": paires.select("id_a", "id_b", "blocs", "score", "decision", "indices")})
    with duckdb.connect(str(base)) as con:  # nouvelles donnees : les anciennes decisions humaines n'ont plus de sens
        con.execute("DROP TABLE IF EXISTS decisions_humaines")
    etape("Construction du référentiel client unique")
    gold = construire_referentiel(base)
    return {"fiches": bronze.height, "clients": gold.height,
            "ambigus": paires.filter(pl.col("decision") == "ambigu").height}


def executer_tout(dossier: Path, base: Path, etape: Etape = lambda s: None) -> dict:
    """Donnees de demonstration (3 fichiers au format connu) : ingestion -> ... -> referentiel."""
    etape("Lecture et alignement des sources")
    bronze = ingestion.ingerer(dossier, base)
    _ecrire(base, {"sources_config": pl.DataFrame(ingestion.config_demo()),
                   "colonnes_extra": pl.DataFrame(schema={"record_id": pl.String, "colonne": pl.String,
                                                          "valeur": pl.String})})
    return _traiter(bronze, base, etape)


def executer_fichiers(fichiers: list[tuple[str, pl.DataFrame, dict[str, str]]], base: Path,
                      etape: Etape = lambda s: None) -> dict:
    """Fichiers quelconques de l'utilisateur, avec la correspondance de colonnes qu'il a validee."""
    etape("Lecture et alignement des fichiers")
    bronze = ingestion.ingerer_fichiers(fichiers, base)
    return _traiter(bronze, base, etape)


CHAMPS_ALIGNABLES = ["id_client", "civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal",
                     "ville", "date_naissance"]


def fichier_corrige(base: Path, source: str, valeurs_referentiel: bool = False) -> pl.DataFrame:
    """Le fichier d'origine d'une source, avec ses colonnes dans le meme ordre, les valeurs corrigees
    et, en premiere colonne, le client_id du referentiel : chaque service peut le reimporter tel quel."""
    silver = lire(base, "silver_clients").filter(pl.col("source") == source)
    if valeurs_referentiel and "correspondance_clients" in _tables(base):
        # Chaque service recoit les valeurs de la FICHE UNIQUE (meme orthographe, meme e-mail principal...)
        # au lieu de ses propres valeurs : tous les services partagent alors exactement la meme donnee.
        gold = lire(base, "gold_clients")
        champs = [c for c in CHAMPS_ALIGNABLES if c in gold.columns and c in silver.columns]
        silver = (silver.join(lire(base, "correspondance_clients"), on="record_id", how="left")
                  .join(gold.select("client_id", *[pl.col(c).alias(f"{c}__ref") for c in champs]),
                        on="client_id", how="left")
                  .with_columns([pl.coalesce(f"{c}__ref", c).alias(c) for c in champs])
                  .drop([f"{c}__ref" for c in champs] + ["client_id"]))
    cfg = lire(base, "sources_config").filter(pl.col("source") == source).sort("ordre")
    corresp = lire(base, "correspondance_clients") if "correspondance_clients" in _tables(base) else None
    out = silver.select("record_id")
    if corresp is not None:
        out = out.join(corresp, on="record_id", how="left")
    extras = lire(base, "colonnes_extra")
    larges = (extras.join(silver.select("record_id"), on="record_id").pivot(on="colonne", index="record_id",
                                                                           values="valeur", aggregate_function="first")
              if not extras.is_empty() else pl.DataFrame({"record_id": silver["record_id"]}))
    colonnes = []
    for col, champ in cfg.select("colonne_origine", "champ").iter_rows():
        if champ == "nom_complet":
            v = pl.concat_str([pl.col("prenom"), pl.col("nom")], separator=" ", ignore_nulls=True)
        elif champ in ingestion.LIBELLES_CHAMPS:
            v = pl.col(champ).cast(pl.String)
        else:
            continue
        colonnes.append(v.alias(col))
    valeurs = silver.select("record_id", *colonnes, pl.col("anomalies").list.join(" ; ").alias("anomalies signalées"))
    out = out.join(valeurs, on="record_id", how="left").join(larges, on="record_id", how="left")
    ordre = [c for c in cfg["colonne_origine"].to_list() if c in out.columns]
    debut = ["client_id"] if "client_id" in out.columns else []
    return out.select(*debut, "record_id", *ordre, "anomalies signalées")


def sources(base: Path) -> list[str]:
    return lire(base, "sources_config")["source"].unique(maintain_order=True).to_list() \
        if "sources_config" in _tables(base) else []


def pret(base: Path) -> bool:
    return {"bronze_clients", "silver_clients", "paires", "gold_clients"} <= _tables(base)
