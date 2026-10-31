"""Generation d'un jeu de donnees clients "sale", realiste et reproductible.

Idee : on cree d'abord de VRAIES personnes fictives (la verite), puis on les
disperse dans 3 systemes d'une entreprise (CRM, boutique en ligne, service client).
Chaque systeme a son propre schema et ses propres defauts de saisie, comme en vrai.
On garde a part la table de verite (quel enregistrement = quelle personne) :
c'est elle qui permettra de MESURER la precision du dedoublonnage.

Usage :
    python -m qdc.generation --n 10000 --seed 42 --out data/raw
"""

from __future__ import annotations

import argparse
import random
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from faker import Faker

from qdc.referentiel import COMMUNES


# --- Parametres de "salete" : un seul endroit pour les regler ---------------
@dataclass(frozen=True)
class Config:
    n_personnes: int = 10_000
    seed: int = 42
    # Probabilite qu'une personne soit presente dans chaque systeme
    p_crm: float = 0.80
    p_boutique: float = 0.55
    p_support: float = 0.30
    # Doublons a l'interieur d'un meme systeme (le client s'inscrit 2 fois)
    p_doublon_interne: float = 0.06
    # Defauts de saisie
    p_faute_frappe: float = 0.08
    p_manquant: float = 0.07
    p_aberrant: float = 0.01
    p_email_invalide: float = 0.02
    p_cp_incoherent: float = 0.02
    p_demenagement: float = 0.05  # adresse differente selon la source (donnee perimee)
    # Pieges pour le dedoublonnage
    p_autre_email: float = 0.20   # e-mail pro ou ancien e-mail dans une source
    p_autre_tel: float = 0.12     # ancien numero dans une source
    p_foyer: float = 0.04         # couple : meme adresse, meme telephone fixe (2 personnes DIFFERENTES)



def tirer_commune(rng: random.Random) -> tuple[str, str]:
    cp, ville, _ = rng.choices(COMMUNES, weights=[c[2] for c in COMMUNES])[0]
    return cp, ville

# --- Petites fonctions de corruption -----------------------------------------
def sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def faute_de_frappe(s: str, rng: random.Random) -> str:
    """Une erreur humaine : lettre supprimee, doublee, inversee ou remplacee."""
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice(["suppr", "double", "inverse", "remplace"])
    if op == "suppr":
        return s[:i] + s[i + 1:]
    if op == "double":
        return s[:i] + s[i] + s[i:]
    if op == "inverse":
        return s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
    voisines = "azertyuiopqsdfghjklmwxcvbn"
    return s[:i] + rng.choice(voisines) + s[i + 1:]


def format_telephone(tel: str, rng: random.Random) -> str:
    """Le meme numero ecrit de 5 facons differentes."""
    chiffres = tel  # 10 chiffres, ex : 0612345678
    paires = " ".join(chiffres[i:i + 2] for i in range(0, 10, 2))
    return rng.choice([
        chiffres,
        paires,
        paires.replace(" ", "."),
        "+33" + chiffres[1:],
        "+33 " + paires[1:],
    ])


def format_date(d: date, style: str) -> str:
    return d.strftime("%d/%m/%Y") if style == "fr" else d.isoformat()


# --- Etape 1 : les vraies personnes ------------------------------------------
def email_unique(base: str, domaine: str, pris: set[str], rng: random.Random) -> str:
    """Dans la realite, une adresse e-mail appartient a UNE personne : deux homonymes
    (deux « Jean Martin ») ne peuvent pas avoir la meme. On ajoute un numero si besoin."""
    email = f"{base}@{domaine}"
    while email in pris:
        email = f"{base}{rng.randint(1, 999)}@{domaine}"
    pris.add(email)
    return email


def generer_personnes(cfg: Config, fake: Faker, rng: random.Random) -> list[dict]:
    personnes = []
    pris: set[str] = set()
    for pid in range(1, cfg.n_personnes + 1):
        femme = rng.random() < 0.5
        prenom = fake.first_name_female() if femme else fake.first_name_male()
        nom = fake.last_name()
        domaine = rng.choice(["gmail.com", "yahoo.fr", "hotmail.fr", "orange.fr", "outlook.fr", "free.fr"])
        base_mail = sans_accents(f"{prenom}.{nom}").lower().replace(" ", "").replace("'", "")
        suffixe = str(rng.randint(1, 99)) if rng.random() < 0.3 else ""
        personnes.append({
            "person_id": pid,
            "civilite": "Mme" if femme else "M.",
            "prenom": prenom,
            "nom": nom,
            "date_naissance": fake.date_of_birth(minimum_age=18, maximum_age=85),
            "email": email_unique(base_mail + suffixe, domaine, pris, rng),
            "telephone": "0" + rng.choice("67") + "".join(rng.choices("0123456789", k=8)),
            "adresse": fake.street_address(),
            "code_postal": None,
            "ville": None,
            "date_inscription": fake.date_between(start_date=date(2018, 1, 1), end_date=date(2026, 9, 30)),
        })
        p = personnes[-1]
        p["code_postal"], p["ville"] = tirer_commune(rng)
        # Identifiants secondaires : un client n'a pas toujours le meme e-mail / numero partout
        pro = rng.choice(["entreprise.fr", "groupe-conseil.com", "univ-paris.fr", "laposte.net"])
        base_pro = sans_accents(p["prenom"][0] + p["nom"]).lower().replace(" ", "").replace("'", "")
        p["email_alt"] = email_unique(base_pro, pro, pris, rng)
        p["tel_alt"] = "0" + rng.choice("67") + "".join(rng.choices("0123456789", k=8))

    # Foyers : certaines personnes vivent avec une autre et partagent le fixe et l'adresse.
    # Ce sont des personnes DIFFERENTES : une regle "meme telephone = meme client" se trompera.
    for p in personnes:
        if rng.random() < cfg.p_foyer:
            conjoint = rng.choice(personnes)
            if conjoint is p:
                continue
            fixe = "0" + rng.choice("12345") + "".join(rng.choices("0123456789", k=8))
            p["tel_alt"] = conjoint["tel_alt"] = fixe
            conjoint["nom"] = p["nom"] if rng.random() < 0.6 else conjoint["nom"]
            for c in ("adresse", "code_postal", "ville"):
                conjoint[c] = p[c]
    return personnes


# --- Etape 2 : une version "salie" d'une personne, vue par un systeme ---------
def salir(p: dict, cfg: Config, fake: Faker, rng: random.Random) -> dict:
    r = dict(p)
    # Fautes de frappe sur le nom ou le prenom
    if rng.random() < cfg.p_faute_frappe:
        champ = rng.choice(["nom", "prenom"])
        r[champ] = faute_de_frappe(r[champ], rng)
    # Casse et accents incoherents
    if rng.random() < 0.15:
        r["nom"] = r["nom"].upper()
    if rng.random() < 0.10:
        r["prenom"], r["nom"] = sans_accents(r["prenom"]), sans_accents(r["nom"])
    # Prenom abrege
    if rng.random() < 0.05:
        r["prenom"] = r["prenom"][0] + "."
    # Identifiants secondaires (autre e-mail, ancien numero ou fixe du foyer)
    if rng.random() < cfg.p_autre_email:
        r["email"] = r["email_alt"]
    if rng.random() < cfg.p_autre_tel:
        r["telephone"] = r["tel_alt"]
    # E-mail : majuscules, espaces, ou invalide
    if rng.random() < 0.10:
        r["email"] = r["email"].capitalize()
    if rng.random() < 0.05:
        r["email"] = " " + r["email"] + " "
    if rng.random() < cfg.p_email_invalide:
        r["email"] = r["email"].replace("@", rng.choice(["", " @", "@@"]))
    # Demenagement : l'adresse n'a pas ete mise a jour partout
    if rng.random() < cfg.p_demenagement:
        r["adresse"] = fake.street_address()
        r["code_postal"], r["ville"] = tirer_commune(rng)
    # Code postal qui ne correspond pas a la ville
    if rng.random() < cfg.p_cp_incoherent:
        r["code_postal"] = tirer_commune(rng)[0]  # un vrai CP, mais pas celui de la ville
    # Valeurs aberrantes
    if rng.random() < cfg.p_aberrant:
        r["date_naissance"] = rng.choice([date(1900, 1, 1), date(2031, 5, 4), date(1867, 3, 12)])
    # Valeurs manquantes
    for champ in ["email", "telephone", "adresse", "date_naissance", "civilite"]:
        if rng.random() < cfg.p_manquant:
            r[champ] = None
    return r


# --- Etape 3 : chaque systeme a son schema et ses habitudes -------------------
def vers_crm(r: dict, rng: random.Random) -> dict:
    return {
        "id_crm": None,
        "Civilité": r["civilite"],
        "Prénom": r["prenom"],
        "Nom": r["nom"],
        "Email": r["email"],
        "Tél": format_telephone(r["telephone"], rng) if r["telephone"] else None,
        "Adresse": r["adresse"],
        "CP": r["code_postal"],
        "Ville": r["ville"].upper() if r["ville"] else None,
        "Date de naissance": format_date(r["date_naissance"], "fr") if r["date_naissance"] else None,
        "Date création": format_date(r["date_inscription"], "fr"),
    }


def vers_boutique(r: dict, rng: random.Random) -> dict:
    # La boutique stocke le nom complet dans un seul champ, dans un ordre variable
    complet = f"{r['prenom']} {r['nom']}" if rng.random() < 0.7 else f"{r['nom']} {r['prenom']}"
    return {
        "customer_id": None,
        "full_name": complet,
        "mail": r["email"],
        "phone": format_telephone(r["telephone"], rng) if r["telephone"] else None,
        "street": r["adresse"],
        "zip": r["code_postal"],
        "city": r["ville"],
        "birthdate": format_date(r["date_naissance"], "iso") if r["date_naissance"] else None,
        "created_at": format_date(r["date_inscription"] + timedelta(days=rng.randint(0, 400)), "iso"),
    }


def vers_support(r: dict, rng: random.Random) -> dict:
    # Le service client saisit vite, au telephone : peu de champs, beaucoup d'erreurs
    return {
        "ticket_client_id": None,
        "client": f"{r['civilite'] or ''} {r['nom']}".strip(),
        "prenom_client": r["prenom"] if rng.random() < 0.6 else None,
        "email_contact": r["email"],
        "numero": format_telephone(r["telephone"], rng) if r["telephone"] else None,
        "code_postal": r["code_postal"],
        "dernier_contact": format_date(date(2026, 10, 1) - timedelta(days=rng.randint(0, 1500)), "iso"),
    }


SYSTEMES = {
    "crm": ("id_crm", "CRM", vers_crm),
    "boutique": ("customer_id", "BTQ", vers_boutique),
    "support": ("ticket_client_id", "SUP", vers_support),
}


# --- Orchestration -----------------------------------------------------------
def generer(cfg: Config, out: Path) -> dict[str, int]:
    rng = random.Random(cfg.seed)
    fake = Faker("fr_FR")
    Faker.seed(cfg.seed)

    personnes = generer_personnes(cfg, fake, rng)
    proba = {"crm": cfg.p_crm, "boutique": cfg.p_boutique, "support": cfg.p_support}
    lignes: dict[str, list[dict]] = {s: [] for s in SYSTEMES}
    verite: list[dict] = []

    for p in personnes:
        presents = [s for s in SYSTEMES if rng.random() < proba[s]] or [rng.choice(list(SYSTEMES))]
        for s in presents:
            n_copies = 2 if rng.random() < cfg.p_doublon_interne else 1
            for _ in range(n_copies):
                col_id, prefixe, convertir = SYSTEMES[s]
                ligne = convertir(salir(p, cfg, fake, rng), rng)
                ligne[col_id] = f"{prefixe}-{len(lignes[s]) + 1:07d}"
                lignes[s].append(ligne)
                verite.append({"source": s, "record_id": ligne[col_id], "person_id": p["person_id"]})

    out.mkdir(parents=True, exist_ok=True)
    for s in SYSTEMES:
        rng.shuffle(lignes[s])  # l'ordre ne doit rien reveler
    pl.DataFrame(lignes["crm"], infer_schema_length=None).write_csv(out / "crm_clients.csv", separator=";")
    pl.DataFrame(lignes["boutique"], infer_schema_length=None).write_excel(out / "boutique_clients.xlsx")
    pl.DataFrame(lignes["support"], infer_schema_length=None).write_parquet(out / "support_clients.parquet")

    # La verite et les personnes d'origine : jamais montrees a l'outil, seulement a l'evaluation
    verite_dir = out.parent / "verite"
    verite_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(verite).write_csv(verite_dir / "verite.csv")
    pl.DataFrame(personnes).write_parquet(verite_dir / "personnes.parquet")

    return {s: len(v) for s, v in lignes.items()} | {"personnes": len(personnes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere un jeu de donnees clients sale et sa verite.")
    parser.add_argument("--n", type=int, default=10_000, help="nombre de personnes reelles")
    parser.add_argument("--seed", type=int, default=42, help="graine : meme graine = memes donnees")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    stats = generer(Config(n_personnes=args.n, seed=args.seed), args.out)
    for k, v in stats.items():
        print(f"{k:>10} : {v:>8,} lignes".replace(",", " "))


if __name__ == "__main__":
    main()
