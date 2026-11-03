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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

# --- Le schema commun (le "contrat" de donnees) ------------------------------
SCHEMA_COMMUN = [
    "record_id", "source", "civilite", "prenom", "nom", "nom_complet", "email",
    "telephone", "adresse", "code_postal", "ville", "date_naissance",
    "date_creation", "date_dernier_contact", "id_client",
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


# =====================================================================================
# Fichiers quelconques : correspondance des colonnes devinee, corrigeable par l'utilisateur
# =====================================================================================
LIBELLES_CHAMPS = {
    "civilite": "Civilité", "prenom": "Prénom", "nom": "Nom", "nom_complet": "Nom complet (prénom + nom)",
    "email": "E-mail", "telephone": "Téléphone", "adresse": "Adresse", "code_postal": "Code postal",
    "ville": "Ville", "date_naissance": "Date de naissance", "date_creation": "Date de création",
    "date_dernier_contact": "Date de dernière activité",
    "id_client": "Identifiant client commun",
}
CONSERVER = "(conserver telle quelle)"
IGNORER = "(ignorer)"

# Noms de colonnes frequents, sans accents, espaces ni ponctuation. L'ORDRE compte :
# « prenom » est teste avant « nom », sinon « prenom » serait reconnu comme un nom.
SYNONYMES = {
    "prenom": ["prenom", "firstname", "givenname", "prenomclient", "prenomcontact"],
    "nom_complet": ["nomcomplet", "fullname", "nometprenom", "prenometnom", "name", "client", "customer",
                    "contact", "nomclient", "customername"],
    "nom": ["nom", "lastname", "surname", "nomdefamille", "familyname", "nomfamille"],
    "civilite": ["civilite", "titre", "title", "salutation", "genre"],
    "email": ["email", "mail", "courriel", "emailcontact", "adressemail", "adresseemail", "emailaddress"],
    "telephone": ["telephone", "tel", "phone", "mobile", "portable", "gsm", "numero", "numtel",
                  "phonenumber", "telephoneportable", "telportable"],
    "code_postal": ["codepostal", "cp", "zip", "zipcode", "postalcode", "postcode"],
    "ville": ["ville", "city", "commune", "localite", "town"],
    "adresse": ["adresse", "address", "rue", "street", "adresse1", "adresseligne1", "voie"],
    "date_naissance": ["datenaissance", "datedenaissance", "naissance", "birthdate", "dateofbirth", "dob",
                       "birthday"],
    "date_creation": ["datecreation", "datedecreation", "created", "createdat", "dateinscription",
                      "inscription", "creationdate", "signupdate"],
    "date_dernier_contact": ["derniercontact", "dernieractivite", "lastcontact", "lastactivity", "updatedat",
                             "datemaj", "datedernieractivite", "lastseen", "datederniercontact"],
}


def _cle(nom: str) -> str:
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", nom.lower()) if unicodedata.category(c) != "Mn")
    return "".join(c for c in t if c.isalnum())


def _par_contenu(serie: pl.Series) -> str | None:
    """Quand le nom de la colonne ne dit rien (« col_7 »), on regarde ce qu'elle contient."""
    s = serie.drop_nulls().cast(pl.String).str.strip_chars()
    s = s.filter(s != "").head(500)
    if s.len() < 3:
        return None
    if s.str.contains(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$").mean() > 0.8:
        return "email"
    if s.str.replace_all(r"[\s.\-()]", "").str.contains(r"^(\+33|0033|0)[1-9]\d{8}$").mean() > 0.8:
        return "telephone"
    return None


def est_identifiant(nom: str) -> bool:
    """« customer_id », « N° client », « ref_commande », « uuid »... : un identifiant technique, pas un nom."""
    n = nom.lower().strip()
    return bool(re.search(r"(^id\b|[_\s-]id$|^id[_\s-]|\bid$|^n°|^no\b|^num(ero)?[_\s]*(client|commande|dossier)"
                          r"|^ref\b|^ref[_\s-]|reference|identifiant|uuid|^code[_\s-]?client)", n))


def deviner_correspondance(df: pl.DataFrame) -> dict[str, str]:
    """Pour chaque colonne : le champ standard qu'elle represente, ou CONSERVER.
    1. les identifiants techniques sont conserves tels quels ;
    2. passage « exact » sur TOUTES les colonnes (« full_name » doit gagner sur « customer_id ») ;
    3. passage « contient » (« telephone_portable » -> telephone) ;
    4. pour les colonnes restantes, on regarde le contenu (une colonne pleine de @ = e-mail).
    Chaque champ n'est attribue qu'une fois."""
    pris: set[str] = set()
    resultat: dict[str, str] = {c: CONSERVER for c in df.columns}
    candidates = [c for c in df.columns if not est_identifiant(c)]

    def attribuer(test) -> None:
        for col in candidates:
            if resultat[col] != CONSERVER:
                continue
            for champ, mots in SYNONYMES.items():
                if champ not in pris and test(_cle(col), mots):
                    resultat[col] = champ
                    pris.add(champ)
                    break

    attribuer(lambda k, mots: k in mots)
    attribuer(lambda k, mots: len(k) > 3 and any(m in k for m in mots if len(m) >= 4))
    for col in candidates:
        if resultat[col] == CONSERVER:
            c = _par_contenu(df[col])
            if c and c not in pris:
                resultat[col] = c
                pris.add(c)
    # « Nom complet » est inutile si prenom ET nom sont deja trouves separement
    if {"prenom", "nom"} <= pris and "nom_complet" in pris:
        resultat = {c: (CONSERVER if v == "nom_complet" else v) for c, v in resultat.items()}
    return resultat


LIMITE_FICHIERS = 10
LIMITE_LIGNES = 200_000   # version en ligne : environ 1 Go de memoire disponible


def ingerer_fichiers(fichiers: list[tuple[str, pl.DataFrame, dict[str, str]]], base: Path) -> pl.DataFrame:
    """fichiers : liste de (nom de la source, tableau, correspondance colonne -> champ).

    Ecrit dans la base :
    - bronze_clients : les champs standards, alignes sur le schema commun ;
    - colonnes_extra : les colonnes conservees (montant, IBAN...), au format long (record_id, colonne, valeur) ;
    - sources_config : la correspondance utilisee, pour savoir quels champs chaque source collecte
      et pour reconstruire chaque fichier d'origine corrige.
    """
    if not 1 <= len(fichiers) <= LIMITE_FICHIERS:
        raise ValueError(f"Déposez entre 1 et {LIMITE_FICHIERS} fichiers.")
    total = sum(df.height for _, df, _ in fichiers)
    if total > LIMITE_LIGNES:
        raise ValueError(f"{total:,} lignes au total : la limite est de {LIMITE_LIGNES:,} sur cette version.".replace(",", " "))
    noms = [n for n, _, _ in fichiers]
    if len(set(noms)) != len(noms):
        raise ValueError("Chaque fichier doit avoir un nom de source différent.")

    morceaux, extras, config = [], [], []
    prefixes: set[str] = set()
    for nom, df, corresp in fichiers:
        champs = {c: ch for c, ch in corresp.items() if ch in LIBELLES_CHAMPS}
        if len(set(champs.values())) != len(champs):
            raise ValueError(f"[{nom}] un même champ est attribué à deux colonnes.")
        if not {"nom", "nom_complet", "email", "telephone"} & set(champs.values()):
            raise ValueError(f"[{nom}] il faut au moins un nom, un e-mail ou un téléphone pour reconnaître un client.")
        # Prefixe des identifiants, UNIQUE par source : deux fichiers nommes « jeu_de_donnees_v1 » et
        # « jeu_de_donnees_v2 » donnaient le meme prefixe, donc des identifiants en double (bug corrige).
        racine = "".join(ch for ch in nom.upper() if ch.isalnum())[:6] or "SRC"
        prefixe, k = racine, 2
        while prefixe in prefixes:
            prefixe, k = f"{racine}{k}", k + 1
        prefixes.add(prefixe)
        texte = df.with_columns(pl.all().cast(pl.String)).with_row_index("_n", offset=1).with_columns(
            record_id=pl.lit(prefixe + "-") + pl.col("_n").cast(pl.String).str.zfill(7))
        lf = texte.select("record_id", *[pl.col(c).alias(ch) for c, ch in champs.items()]).with_columns(
            [pl.lit(None, dtype=pl.String).alias(c) for c in SCHEMA_COMMUN
             if c not in champs.values() and c != "record_id"]
        ).with_columns(source=pl.lit(nom))
        # Dates : n'importe quel format courant -> ISO (la validite est controlee au nettoyage)
        from qdc.atelier import FORMATS_DATE
        for d in ("date_naissance", "date_creation", "date_dernier_contact"):
            if d in champs.values():
                t = pl.col(d).str.strip_chars().str.slice(0, 10)
                lf = lf.with_columns(pl.coalesce([t.str.strptime(pl.Date, f, strict=False) for f in FORMATS_DATE])
                                     .cast(pl.String).alias(d))
        morceaux.append(lf.select(SCHEMA_COMMUN).with_columns(pl.lit(nom).alias("_fichier")))
        gardees = [c for c, ch in corresp.items() if ch == CONSERVER]
        if gardees:
            extras.append(texte.select("record_id", *gardees).unpivot(index="record_id", variable_name="colonne",
                                                                    value_name="valeur").drop_nulls("valeur"))
        config += [{"source": nom, "colonne_origine": c, "champ": ch, "ordre": i}
                   for i, (c, ch) in enumerate(corresp.items())]

    bronze = pl.concat(morceaux).with_columns(
        pl.lit(datetime.now(timezone.utc).isoformat(timespec="seconds")).alias("_ingere_le"))
    extra = pl.concat(extras) if extras else pl.DataFrame(schema={"record_id": pl.String, "colonne": pl.String,
                                                                 "valeur": pl.String})
    base.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(base)) as con:
        for table, d in (("bronze_clients", bronze), ("colonnes_extra", extra), ("sources_config", pl.DataFrame(config))):
            con.register("t", d.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM t")
            con.unregister("t")
    return bronze


def config_demo() -> list[dict]:
    """La correspondance des 3 fichiers de demonstration, au meme format que pour des fichiers quelconques."""
    return [{"source": s.nom, "colonne_origine": c, "champ": ch, "ordre": i}
            for s in SOURCES for i, (c, ch) in enumerate(s.correspondance.items()) if ch != "record_id"]


def applicable(base: Path) -> dict[str, set[str]] | None:
    """Champs collectes par chaque source (un champ non collecte est « non applicable », pas « manquant »)."""
    with duckdb.connect(str(base)) as con:
        if "sources_config" not in {t[0] for t in con.execute("SHOW TABLES").fetchall()}:
            return None
        cfg = con.sql("SELECT source, champ FROM sources_config").pl()
    out: dict[str, set[str]] = {}
    for src, ch in cfg.iter_rows():
        if ch in LIBELLES_CHAMPS:
            out.setdefault(src, set()).add(ch)
    return out


def champs_collectes(base: Path) -> set[str] | None:
    """Champs qu'au moins une source fournit (directement, ou par deduction sure :
    un nom complet donne prenom + nom, un code postal donne la ville)."""
    a = applicable(base)
    if a is None:
        return None
    tous = set().union(*a.values()) if a else set()
    if "nom_complet" in tous:
        tous |= {"prenom", "nom"}
    if "code_postal" in tous:
        tous.add("ville")
    return tous


def identifiants_communs(tableaux: list[pl.DataFrame], recouvrement_min: float = 0.3) -> set[str]:
    """Colonnes d'identifiant presentes sous le MEME nom dans au moins deux fichiers, dont les valeurs
    se recoupent (ex. « ID Client » 101, 102... dans le CRM et dans la compta). Si c'est le cas, l'outil
    propose de s'en servir pour reunir les fiches : meme identifiant = meme client.
    (Des identifiants de systemes differents, eux, ne se recoupent pas : ils restent de simples colonnes.)"""
    par_nom: dict[str, list[set]] = {}
    for df in tableaux:
        for col in df.columns:
            if est_identifiant(col):
                par_nom.setdefault(_cle(col), []).append(set(df[col].drop_nulls().cast(pl.String).to_list()))
    communs = set()
    for cle, ensembles in par_nom.items():
        if len(ensembles) >= 2:
            a, b = ensembles[0], ensembles[1]
            if a and b and len(a & b) / min(len(a), len(b)) >= recouvrement_min:
                communs.add(cle)
    return {col for df in tableaux for col in df.columns if _cle(col) in communs}
