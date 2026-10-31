"""Ingestion : lire n'importe quelle source et l'aligner sur un schema commun (couche bronze).

Principes (architecture "medaillon") :
- BRONZE : la donnee brute, alignee sur un schema commun mais NON corrigee.
  Tout est lu en texte : on ne fait confiance a aucun type tant qu'on n'a pas valide.
- SILVER (phase 3) : la donnee nettoyee et validee.
- GOLD (phase 5) : la fiche client unique.

Chaque ligne garde sa source et son identifiant d'origine : on doit toujours
pouvoir remonter a l'enregistrement d'origine (lignage / tracabilite).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

# --- Le schema commun (le "contrat" de donnees) ------------------------------
SCHEMA_COMMUN = [
    "record_id", "source", "civilite", "prenom", "nom", "nom_complet", "email",
    "telephone", "adresse", "code_postal", "ville", "date_naissance",
    "date_creation", "date_dernier_contact",
]


@dataclass(frozen=True)
class Source:
    """Decrit une source : ou la trouver et comment ses colonnes se traduisent."""
    nom: str
    fichier: str
    correspondance: dict[str, str]           # colonne d'origine -> colonne commune
    format_date: str | None = None           # ex : "%d/%m/%Y" ; None = ISO
    extras: dict[str, str] = field(default_factory=dict)


# La correspondance de colonnes est de la CONFIGURATION, pas du code :
# ajouter une nouvelle source = ajouter une entree ici.
SOURCES = [
    Source(
        nom="crm", fichier="crm_clients.csv", format_date="%d/%m/%Y",
        correspondance={
            "id_crm": "record_id", "Civilité": "civilite", "Prénom": "prenom", "Nom": "nom",
            "Email": "email", "Tél": "telephone", "Adresse": "adresse", "CP": "code_postal",
            "Ville": "ville", "Date de naissance": "date_naissance", "Date création": "date_creation",
        },
    ),
    Source(
        nom="boutique", fichier="boutique_clients.xlsx",
        correspondance={
            "customer_id": "record_id", "full_name": "nom_complet", "mail": "email",
            "phone": "telephone", "street": "adresse", "zip": "code_postal", "city": "ville",
            "birthdate": "date_naissance", "created_at": "date_creation",
        },
    ),
    Source(
        nom="support", fichier="support_clients.parquet",
        correspondance={
            "ticket_client_id": "record_id", "client": "nom_complet", "prenom_client": "prenom",
            "email_contact": "email", "numero": "telephone", "code_postal": "code_postal",
            "dernier_contact": "date_dernier_contact",
        },
    ),
]


# --- Lecture generique ---------------------------------------------------------
def detecter_separateur(chemin: Path) -> str:
    with open(chemin, encoding="utf-8", newline="") as f:
        return csv.Sniffer().sniff(f.read(20_000), delimiters=";,\t|").delimiter


def lire_fichier(chemin: Path) -> pl.LazyFrame:
    """Lit CSV, Excel ou Parquet, toutes colonnes en texte.

    CSV et Parquet sont lus en mode "lazy" (paresseux) : Polars ne charge rien tant
    qu'on n'a pas demande le resultat, et peut traiter des fichiers plus gros que la
    memoire. C'est la reponse au probleme de ton ancienne API FastAPI.
    """
    ext = chemin.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pl.scan_csv(chemin, separator=detecter_separateur(chemin), infer_schema=False)
    if ext == ".parquet":
        return pl.scan_parquet(chemin).select(pl.all().cast(pl.String))
    if ext in {".xlsx", ".xls"}:
        # Excel n'est pas un format "streamable" : on le lit en entier (limite connue)
        return pl.read_excel(chemin, infer_schema_length=0).lazy().select(pl.all().cast(pl.String))
    raise ValueError(f"Format non pris en charge : {ext}")


def aligner(lf: pl.LazyFrame, src: Source) -> pl.LazyFrame:
    """Renomme vers le schema commun, ajoute les colonnes manquantes, convertit les dates en ISO."""
    colonnes = lf.collect_schema().names()
    manquantes_source = set(src.correspondance) - set(colonnes)
    if manquantes_source:
        raise ValueError(f"[{src.nom}] colonnes attendues absentes : {sorted(manquantes_source)}")

    lf = lf.select([pl.col(c).alias(dst) for c, dst in src.correspondance.items()])
    presentes = set(src.correspondance.values())
    lf = lf.with_columns(
        [pl.lit(None, dtype=pl.String).alias(c) for c in SCHEMA_COMMUN if c not in presentes]
    ).with_columns(pl.lit(src.nom).alias("source"))

    if src.format_date:  # on harmonise uniquement le FORMAT ; la validite sera controlee ensuite
        dates = [c for c in ("date_naissance", "date_creation", "date_dernier_contact") if c in presentes]
        lf = lf.with_columns([
            pl.col(c).str.strptime(pl.Date, src.format_date, strict=False).cast(pl.String).alias(c)
            for c in dates
        ])
    return lf.select(SCHEMA_COMMUN)


def ingerer(dossier: Path, base: Path, sources: list[Source] = SOURCES) -> pl.DataFrame:
    """Lit toutes les sources, les empile et ecrit la table bronze dans DuckDB."""
    morceaux = []
    for src in sources:
        lf = aligner(lire_fichier(dossier / src.fichier), src)
        morceaux.append(lf.with_columns(pl.lit(src.fichier).alias("_fichier")))
    bronze = pl.concat(morceaux).with_columns(
        pl.lit(datetime.now(timezone.utc).isoformat(timespec="seconds")).alias("_ingere_le")
    ).collect()

    base.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(base)) as con:
        con.register("bronze_df", bronze.to_arrow())
        con.execute("CREATE OR REPLACE TABLE bronze_clients AS SELECT * FROM bronze_df")
    return bronze


if __name__ == "__main__":
    df = ingerer(Path("data/raw"), Path("data/qdc.duckdb"))
    print(df.group_by("source").len().sort("source"))
    print(df.head(5))
