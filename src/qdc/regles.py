"""Regles metier : comment normaliser et valider chaque type de champ.

Un seul endroit pour les regles, utilise a la fois par le DIAGNOSTIC (mesurer)
et par le NETTOYAGE (corriger). Ainsi, on mesure exactement ce qu'on corrige.

Chaque valeur peut etre dans 4 etats :
- "valide"      : deja au format de reference
- "corrigeable" : mauvais format, mais on peut la corriger sans risque (espaces, majuscules...)
- "invalide"    : on ne peut pas la corriger automatiquement (e-mail sans @, date impossible...)
- "manquante"   : vide
"""

from __future__ import annotations

import polars as pl

from qdc.referentiel import VILLE_PAR_CP

RE_EMAIL = r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$"
RE_TEL_REF = r"^\+33[1-9]\d{8}$"            # format de reference : E.164, ex +33612345678
RE_CP = r"^(\d{5}|2[AB]\d{3})$"
AGE_MIN, AGE_MAX = 16, 110


def vide(e: pl.Expr) -> pl.Expr:
    return e.is_null() | (e.str.strip_chars() == "")


# --- Normalisations (transforment sans risque) -----------------------------
def norm_texte(e: pl.Expr) -> pl.Expr:
    """Espaces en trop supprimes ; vide -> null."""
    t = e.str.strip_chars().str.replace_all(r"\s+", " ")
    return pl.when(t == "").then(None).otherwise(t)


def norm_email(e: pl.Expr) -> pl.Expr:
    return norm_texte(e).str.to_lowercase()


def norm_telephone(e: pl.Expr) -> pl.Expr:
    """Tout numero francais -> +33XXXXXXXXX. Renvoie null si ce n'est pas un numero francais."""
    chiffres = e.str.replace_all(r"[^\d+]", "")
    national = (
        pl.when(chiffres.str.starts_with("+33")).then(pl.lit("0") + chiffres.str.slice(3))
        .when(chiffres.str.starts_with("0033")).then(pl.lit("0") + chiffres.str.slice(4))
        .otherwise(chiffres)
    )
    return pl.when(national.str.contains(r"^0[1-9]\d{8}$")).then(pl.lit("+33") + national.str.slice(1))


def norm_cp(e: pl.Expr) -> pl.Expr:
    t = norm_texte(e).str.to_uppercase()
    # Un CP lu comme un nombre perd son zero initial : 6000 -> 06000
    return pl.when(t.str.contains(r"^\d{4}$")).then(pl.lit("0") + t).otherwise(t)


def norm_ville(e: pl.Expr) -> pl.Expr:
    """Cle de comparaison : majuscules, sans accents ni tirets."""
    return (
        norm_texte(e).str.to_uppercase()
        .str.replace_all("[ÀÂÄ]", "A").str.replace_all("[ÉÈÊË]", "E").str.replace_all("[ÎÏ]", "I")
        .str.replace_all("[ÔÖ]", "O").str.replace_all("[ÙÛÜ]", "U").str.replace_all("Ç", "C")
        .str.replace_all(r"[-']", " ")
    )


def en_date(e: pl.Expr) -> pl.Expr:
    return e.str.strptime(pl.Date, "%Y-%m-%d", strict=False)


# --- Statut de chaque valeur -------------------------------------------------
def statut(brut: pl.Expr, est_valide_brut: pl.Expr, est_valide_corrige: pl.Expr) -> pl.Expr:
    return (
        pl.when(vide(brut)).then(pl.lit("manquante"))
        .when(est_valide_brut).then(pl.lit("valide"))
        .when(est_valide_corrige).then(pl.lit("corrigeable"))
        .otherwise(pl.lit("invalide"))
    )


def statut_email(c: str) -> pl.Expr:
    e = pl.col(c)
    return statut(e, e.str.contains(RE_EMAIL), norm_email(e).str.contains(RE_EMAIL))


def statut_telephone(c: str) -> pl.Expr:
    e = pl.col(c)
    return statut(e, e.str.contains(RE_TEL_REF), norm_telephone(e).is_not_null())


def statut_cp(c: str) -> pl.Expr:
    e = pl.col(c)
    return statut(e, e.str.contains(RE_CP), norm_cp(e).str.contains(RE_CP))


def statut_date_naissance(c: str, aujourd_hui: pl.Expr) -> pl.Expr:
    """Une date de naissance doit exister ET etre plausible (age entre 16 et 110 ans)."""
    d = en_date(pl.col(c))
    age = (aujourd_hui - d).dt.total_days() / 365.25
    plausible = d.is_not_null() & age.is_between(AGE_MIN, AGE_MAX)
    return statut(pl.col(c), plausible, plausible)


def cp_coherent(cp: str, ville: str) -> pl.Expr:
    """Le CP correspond-il a la ville ? null si on ne peut pas verifier (CP inconnu du referentiel)."""
    attendue = norm_cp(pl.col(cp)).replace_strict(
        {k: v for k, v in VILLE_PAR_CP.items()}, default=None, return_dtype=pl.String
    )
    return pl.when(attendue.is_null() | vide(pl.col(ville))).then(None).otherwise(
        norm_ville(attendue) == norm_ville(pl.col(ville))
    )
