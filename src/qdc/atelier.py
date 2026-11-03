"""Mode 1 : l'atelier de nettoyage d'un fichier quelconque (clients, ventes, comptabilite, Kaggle...).

Principe central : une operation est une DONNEE, par exemple
    {"op": "remplir_vides", "colonne": "montant", "methode": "mediane"}
La liste des operations forme une RECETTE. On obtient ainsi gratuitement :
- le journal (la recette, avec le nombre de valeurs touchees par chaque operation) ;
- l'annulation (on rejoue la recette sans sa derniere operation, a partir du fichier d'origine) ;
- la reutilisation : le mois suivant, on applique la meme recette au nouvel export.

Le fichier d'origine n'est jamais modifie. Chaque operation declare les types de colonnes
compatibles : l'interface ne peut pas proposer une moyenne sur une colonne de noms.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

NUMERIQUES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
              pl.Float32, pl.Float64)


# =====================================================================================
# 1. Chargement
# =====================================================================================
def charger(contenu: bytes, nom: str) -> pl.DataFrame:
    """Lit un CSV (separateur et encodage detectes), un Excel ou un Parquet."""
    ext = Path(nom).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(io.BytesIO(contenu))
    if ext in {".xlsx", ".xls"}:
        return pl.read_excel(io.BytesIO(contenu))
    if ext in {".csv", ".txt"}:
        for encodage in ("utf-8", "cp1252", "latin-1"):  # Excel francais enregistre souvent en cp1252
            try:
                texte = contenu.decode(encodage)
                break
            except UnicodeDecodeError:
                continue
        try:
            sep = csv.Sniffer().sniff(texte[:20_000], delimiters=";,\t|").delimiter
        except csv.Error:
            sep = ","
        lire = lambda dates: pl.read_csv(io.StringIO(texte), separator=sep, infer_schema_length=10_000,  # noqa: E731
                                         try_parse_dates=dates, truncate_ragged_lines=True)
        try:
            return lire(True)
        except pl.exceptions.ComputeError:
            # Dates en plusieurs formats dans une meme colonne (« 03/02/2025 », « 2025-02-04 ») :
            # on les garde en texte, et l'atelier proposera de les convertir.
            return lire(False)
    raise ValueError(f"Format non pris en charge : {ext} (formats acceptes : CSV, Excel, Parquet)")


# =====================================================================================
# 2. Diagnostic (profil des colonnes)
# =====================================================================================
RE_EMAIL = r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"
RE_TEL = r"^\+?[\d .\-()]{9,}$"
RE_NOMBRE_FR = r"^-?[\d  ]+(,\d+)?\s*[€%]?$"      # « 1 234,56 », « 12,5 % »
RE_DATE = r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"


def _part(serie: pl.Series, motif: str) -> float:
    s = serie.drop_nulls().cast(pl.String).str.strip_chars()
    s = s.filter(s != "")
    return float(s.str.contains(motif).mean()) if s.len() else 0.0


def type_colonne(serie: pl.Series) -> str:
    """Type « metier » d'une colonne : ce qu'elle contient vraiment, pas seulement son type technique."""
    if serie.dtype in NUMERIQUES:
        return "nombre"
    if serie.dtype in (pl.Date, pl.Datetime):
        return "date"
    if serie.dtype == pl.Boolean:
        return "booleen"
    if serie.dtype != pl.String:
        return "autre"
    if _part(serie, RE_EMAIL) > 0.8:
        return "email"
    if _part(serie, RE_DATE) > 0.8:
        return "date (texte)"
    if _part(serie, r"^-?\d+([.,]\d+)?$") > 0.9 or _part(serie, RE_NOMBRE_FR) > 0.9:
        return "nombre (texte)"
    if _part(serie, RE_TEL) > 0.8:
        return "telephone"
    return "texte"


def bornes_iqr(serie: pl.Series, k: float = 1.5) -> tuple[float, float] | None:
    """Methode de l'ecart interquartile : est aberrant ce qui sort de [Q1 - k.IQR ; Q3 + k.IQR].
    Robuste : contrairement a la moyenne, les quartiles ne sont pas tires par les valeurs extremes."""
    s = serie.drop_nulls().drop_nans() if serie.dtype in (pl.Float32, pl.Float64) else serie.drop_nulls()
    if s.len() < 8:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - k * iqr, q3 + k * iqr


@dataclass
class ProfilColonne:
    nom: str
    type: str
    n_vides: int
    pct_vides: float
    n_distincts: int
    exemples: list[str]
    stats: dict = field(default_factory=dict)     # nombres : min, max, moyenne, mediane, asymetrie
    n_aberrantes: int = 0
    n_espaces: int = 0                            # textes : valeurs avec espaces en trop


def vides(serie: pl.Series) -> pl.Series:
    v = serie.is_null()
    if serie.dtype == pl.String:
        v = v | (serie.str.strip_chars() == "")
    if serie.dtype in (pl.Float32, pl.Float64):
        v = v | serie.is_nan()
    return v.fill_null(True)


def profiler(df: pl.DataFrame) -> tuple[list[ProfilColonne], int]:
    """Profil de chaque colonne + nombre de lignes en double exact."""
    profils = []
    for nom in df.columns:
        s = df[nom]
        n_vides = int(vides(s).sum())
        p = ProfilColonne(nom=nom, type=type_colonne(s), n_vides=n_vides,
                          pct_vides=n_vides / df.height if df.height else 0.0,
                          n_distincts=s.n_unique(),
                          exemples=[str(x) for x in s.drop_nulls().unique(maintain_order=True).head(3).to_list()])
        if p.type == "nombre":
            ok = s.drop_nulls()
            if ok.dtype in (pl.Float32, pl.Float64):
                ok = ok.drop_nans()
            if ok.len():
                moy, med, ect = ok.mean(), ok.median(), ok.std()
                p.stats = {"min": ok.min(), "max": ok.max(), "moyenne": moy, "mediane": med,
                           "asymetrie": float(ok.skew()) if ok.len() > 2 and ect else 0.0}
                b = bornes_iqr(ok)
                if b:
                    p.n_aberrantes = int(((ok < b[0]) | (ok > b[1])).sum())
        elif s.dtype == pl.String:
            ok = s.drop_nulls()
            p.n_espaces = int((ok != ok.str.strip_chars().str.replace_all(r"\s+", " ")).sum())
        profils.append(p)
    return profils, df.height - df.unique().height


# =====================================================================================
# 3. Operations
# =====================================================================================
@dataclass
class Resultat:
    df: pl.DataFrame
    n: int           # nombre de lignes ou de valeurs touchees
    message: str


def _verifier_colonne(df: pl.DataFrame, c: str) -> None:
    if c not in df.columns:
        raise ValueError(f"Colonne introuvable : « {c} »")


def op_supprimer_doublons(df: pl.DataFrame, colonnes: list[str] | None = None) -> Resultat:
    """Garde la premiere occurrence. Sur toutes les colonnes, ou seulement sur celles choisies
    (ex. un meme numero de facture saisi deux fois avec une date differente)."""
    sub = colonnes or None
    for c in colonnes or []:
        _verifier_colonne(df, c)
    out = df.unique(subset=sub, keep="first", maintain_order=True)
    cible = "toutes les colonnes" if not colonnes else ", ".join(colonnes)
    return Resultat(out, df.height - out.height, f"Doublons supprimés ({cible})")


def op_remplir_vides(df: pl.DataFrame, colonne: str, methode: str, valeur=None) -> Resultat:
    """methode : mediane | moyenne | plus_frequente | valeur."""
    _verifier_colonne(df, colonne)
    s = df[colonne]
    masque = vides(s)
    n = int(masque.sum())
    if methode in ("mediane", "moyenne"):
        if s.dtype not in NUMERIQUES:
            raise ValueError(f"La {methode} n'a de sens que pour une colonne numérique (« {colonne} » ne l'est pas).")
        ok = s.filter(~masque)
        remplacement = ok.median() if methode == "mediane" else ok.mean()
        if s.dtype not in (pl.Float32, pl.Float64):
            remplacement = round(remplacement)
        else:
            # Meme precision que les autres valeurs de la colonne : un montant a 2 decimales
            # reste a 2 decimales (69.84, et non 69.83500000000001).
            decimales = int(ok.drop_nans().cast(pl.String).str.split(".").list.get(1, null_on_oob=True)
                            .str.len_chars().fill_null(0).max() or 0)
            remplacement = round(remplacement, min(decimales, 6))
    elif methode == "plus_frequente":
        remplacement = s.filter(~masque).mode().sort().first()
    elif methode == "valeur":
        if valeur is None or str(valeur) == "":
            raise ValueError("Indiquez la valeur de remplacement.")
        remplacement = valeur
    else:
        raise ValueError(f"Méthode inconnue : {methode}")
    try:
        rempl = pl.lit(remplacement).cast(s.dtype, strict=True)
    except Exception as e:
        raise ValueError(f"« {remplacement} » n'est pas compatible avec la colonne « {colonne} ».") from e
    out = df.with_columns(pl.when(pl.Series(masque)).then(rempl).otherwise(pl.col(colonne)).alias(colonne))
    libelle = {"mediane": "la médiane", "moyenne": "la moyenne", "plus_frequente": "la valeur la plus fréquente",
               "valeur": "une valeur choisie"}[methode]
    affiche = f"{remplacement:.2f}" if isinstance(remplacement, float) else str(remplacement)
    return Resultat(out, n, f"Vides de « {colonne} » remplacés par {libelle} ({affiche})")


def op_supprimer_lignes_vides(df: pl.DataFrame, colonnes: list[str]) -> Resultat:
    for c in colonnes:
        _verifier_colonne(df, c)
    masque = pl.Series([False] * df.height)
    for c in colonnes:
        masque = masque | vides(df[c])
    out = df.filter(~masque)
    return Resultat(out, df.height - out.height, f"Lignes supprimées quand vide : {', '.join(colonnes)}")


def op_traiter_aberrantes(df: pl.DataFrame, colonne: str, action: str, k: float = 1.5) -> Resultat:
    """Valeurs INHABITUELLES (methode statistique de l'ecart interquartile).
    action : signaler (colonne d'alerte) | vider | plafonner (ramene aux bornes) | supprimer (les lignes)."""
    _verifier_colonne(df, colonne)
    s = df[colonne]
    if s.dtype not in NUMERIQUES:
        raise ValueError(f"« {colonne} » n'est pas numérique : impossible de calculer des valeurs aberrantes.")
    b = bornes_iqr(s, k)
    if b is None:
        return Resultat(df, 0, f"Aucune valeur aberrante détectable dans « {colonne} »")
    bas, haut = b
    hors = ((pl.col(colonne) < bas) | (pl.col(colonne) > haut)).fill_null(False)
    n = int(df.select(hors.sum()).item())
    if action == "signaler":
        out = df.with_columns(hors.alias(f"{colonne}_aberrante"))
        msg = f"Valeurs aberrantes de « {colonne} » signalées (colonne « {colonne}_aberrante »)"
    elif action == "vider":
        out = df.with_columns(pl.when(hors).then(None).otherwise(pl.col(colonne)).alias(colonne))
        msg = f"Valeurs inhabituelles de « {colonne} » vidées"
    elif action == "plafonner":
        borne = (lambda x: round(x)) if s.dtype not in (pl.Float32, pl.Float64) else (lambda x: x)
        out = df.with_columns(pl.col(colonne).clip(borne(bas), borne(haut)).cast(s.dtype))
        msg = f"Valeurs de « {colonne} » ramenées entre {bas:.2f} et {haut:.2f}"
    elif action == "supprimer":
        out = df.filter(~hors)
        msg = f"Lignes avec une valeur aberrante dans « {colonne} » supprimées"
    else:
        raise ValueError(f"Action inconnue : {action}")
    return Resultat(out, n, msg)


def op_nettoyer_texte(df: pl.DataFrame, colonnes: list[str], casse: str = "aucune") -> Resultat:
    """Espaces en trop retires ; casse : aucune | majuscules | minuscules | titre."""
    n = 0
    exprs = []
    for c in colonnes:
        _verifier_colonne(df, c)
        if df[c].dtype != pl.String:
            raise ValueError(f"« {c} » n'est pas une colonne de texte.")
        e = pl.col(c).str.strip_chars().str.replace_all(r"\s+", " ")
        e = {"aucune": e, "majuscules": e.str.to_uppercase(), "minuscules": e.str.to_lowercase(),
             "titre": e.str.to_titlecase()}[casse]
        exprs.append(e.alias(c))
        n += int(df.select((pl.col(c) != e).fill_null(False).sum()).item())
    out = df.with_columns(exprs)
    detail = "" if casse == "aucune" else f", casse : {casse}"
    return Resultat(out, n, f"Texte nettoyé (espaces{detail}) : {', '.join(colonnes)}")


def texte_vers_nombre(e: pl.Expr) -> pl.Expr:
    """« 1 234,56 € » -> 1234.56 ; « 12,5 % » -> 12.5 ; « 1.234,56 » -> 1234.56 ; « 1,234.56 » -> 1234.56."""
    t = e.cast(pl.String).str.strip_chars().str.replace_all(r"[€$£%\s\u00a0\u202f]", "")
    anglais = t.str.contains(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")            # 1,234.56
    francais_points = t.str.contains(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")    # 1.234,56
    t = (pl.when(anglais).then(t.str.replace_all(",", ""))
         .when(francais_points).then(t.str.replace_all(r"\.", "").str.replace(",", "."))
         .otherwise(t.str.replace(",", ".")))
    return t.cast(pl.Float64, strict=False)


def op_convertir_nombre(df: pl.DataFrame, colonne: str) -> Resultat:
    _verifier_colonne(df, colonne)
    avant_vides = int(vides(df[colonne]).sum())
    out = df.with_columns(texte_vers_nombre(pl.col(colonne)).alias(colonne))
    echecs = int(out[colonne].is_null().sum()) - avant_vides
    msg = f"« {colonne} » converti en nombre"
    if echecs:
        msg += f" ({echecs} valeurs illisibles devenues vides)"
    return Resultat(out, df.height - avant_vides, msg)


FORMATS_DATE = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%y", "%m/%d/%Y"]


def op_convertir_date(df: pl.DataFrame, colonne: str) -> Resultat:
    """Essaie plusieurs formats ; garde, pour chaque valeur, le premier qui fonctionne.
    (Le jour avant le mois est essaye en premier : format francais.)"""
    _verifier_colonne(df, colonne)
    t = pl.col(colonne).cast(pl.String).str.strip_chars().str.slice(0, 10)
    out = df.with_columns(pl.coalesce([t.str.strptime(pl.Date, f, strict=False) for f in FORMATS_DATE]).alias(colonne))
    avant = int((~vides(df[colonne])).sum())
    echecs = avant - int(out[colonne].is_not_null().sum())
    msg = f"« {colonne} » converti en date"
    if echecs:
        msg += f" ({echecs} valeurs illisibles devenues vides)"
    return Resultat(out, avant - echecs, msg)


def op_renommer(df: pl.DataFrame, colonne: str, nouveau: str) -> Resultat:
    _verifier_colonne(df, colonne)
    if not nouveau or nouveau in df.columns:
        raise ValueError(f"Nom invalide ou déjà utilisé : « {nouveau} »")
    return Resultat(df.rename({colonne: nouveau}), 1, f"Colonne « {colonne} » renommée en « {nouveau} »")


def op_supprimer_colonnes(df: pl.DataFrame, colonnes: list[str]) -> Resultat:
    for c in colonnes:
        _verifier_colonne(df, c)
    return Resultat(df.drop(colonnes), len(colonnes), f"Colonnes supprimées : {', '.join(colonnes)}")


def op_remplacer(df: pl.DataFrame, colonne: str, rechercher: str, remplacer: str) -> Resultat:
    """Remplace une valeur EXACTE (ex. « N/A », « inconnu », « - ») : pas de remplacement partiel cache."""
    _verifier_colonne(df, colonne)
    s = df[colonne].cast(pl.String)
    masque = (s.str.strip_chars() == rechercher.strip()).fill_null(False)
    nouvelle = None if remplacer == "" else remplacer
    out = df.with_columns(pl.when(masque).then(pl.lit(nouvelle, dtype=pl.String))
                          .otherwise(pl.col(colonne).cast(pl.String)).alias(colonne))
    if df[colonne].dtype != pl.String:  # on retrouve le type d'origine si possible
        try:
            out = out.with_columns(pl.col(colonne).cast(df[colonne].dtype))
        except Exception:
            pass
    cible = "vide" if nouvelle is None else f"« {remplacer} »"
    return Resultat(out, int(masque.sum()), f"« {rechercher} » remplacé par {cible} dans « {colonne} »")


# =====================================================================================
# Regles de bon sens : « impossible » n'est pas « inhabituel »
# =====================================================================================
# La statistique dit « inhabituel par rapport aux autres » ; elle ne sait pas qu'un age ne peut pas
# depasser 120 ans, ni qu'un panier ne peut pas etre negatif. Ces regles le disent. Elles viennent
# de l'IA (si une cle est configuree) ou de cette liste ; l'utilisateur les valide ou les modifie,
# et c'est le CODE qui les applique.
REGLES_CONNUES = [
    (("age",), 0, 120, "un âge est compris entre 0 et 120 ans"),
    (("pourcentage", "taux", "pct", "remise"), 0, 100, "un pourcentage est compris entre 0 et 100"),
    (("note", "score", "satisfaction"), 0, None, "une note n'est pas négative"),
    (("quantite", "qte", "qty", "nombre", "nb"), 0, None, "une quantité n'est pas négative"),
    (("panier", "prix", "price", "tarif", "cout", "montantachat", "ca", "chiffre"), 0, None,
     "un prix ou un panier n'est pas négatif"),
    (("annee", "year"), 1900, 2100, "une année est comprise entre 1900 et 2100"),
]


def _mots(nom: str) -> list[str]:
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", nom.lower()) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", t)


def _n(x) -> str:
    """120.0 -> « 120 » ; 12.5 -> « 12,5 »."""
    return f"{x:g}".replace(".", ",") if isinstance(x, (int, float)) else str(x)


def regles_par_defaut(df: pl.DataFrame) -> dict[str, dict]:
    """Regles connues, reconnues au nom de la colonne (« Age client » -> age entre 0 et 120)."""
    regles = {}
    for col in df.columns:
        if type_colonne(df[col]) not in ("nombre", "nombre (texte)"):
            continue
        mots = _mots(col)
        colle = "".join(mots)
        for cles, mini, maxi, raison in REGLES_CONNUES:
            if any(c in mots or (len(c) > 4 and c in colle) for c in cles):
                regles[col] = {"min": mini, "max": maxi, "raison": raison, "source": "règle connue"}
                break
    return regles


PROMPT_REGLES = """Tu es expert en qualite des donnees. Pour chaque colonne NUMERIQUE d'un tableau, indique les
valeurs POSSIBLES (pas habituelles : possibles). Exemple : un age entre 0 et 120, un panier >= 0,
un pourcentage entre 0 et 100. Si tu ne sais pas, n'invente rien : mets null.
Reponds UNIQUEMENT en JSON : {"regles": [{"colonne": "...", "min": nombre ou null, "max": nombre ou null,
"raison": "phrase courte en francais"}]}"""


def regles_ia(df: pl.DataFrame, appeler) -> dict[str, dict]:
    """L'IA propose des regles a partir des noms de colonnes et de quelques exemples.
    Garde-fous : seules les colonnes numeriques existantes sont acceptees, bornes numeriques, min <= max."""
    import json
    cols = [c for c in df.columns if type_colonne(df[c]) in ("nombre", "nombre (texte)")]
    if not cols:
        return {}
    apercu = [{"colonne": c, "exemples": [str(x) for x in df[c].drop_nulls().head(6).to_list()]} for c in cols]
    brut = json.loads(appeler([{"role": "system", "content": PROMPT_REGLES},
                               {"role": "user", "content": json.dumps(apercu, ensure_ascii=False)}]))
    regles = {}
    for r in (brut.get("regles", []) if isinstance(brut, dict) else []):
        if not isinstance(r, dict) or r.get("colonne") not in cols:
            continue
        mini, maxi = r.get("min"), r.get("max")
        if not all(v is None or isinstance(v, (int, float)) for v in (mini, maxi)) or (mini is None and maxi is None):
            continue
        if mini is not None and maxi is not None and mini > maxi:
            continue
        regles[r["colonne"]] = {"min": mini, "max": maxi, "raison": str(r.get("raison", ""))[:120], "source": "IA"}
    return regles


def _hors_plage(colonne: str, mini, maxi) -> pl.Expr:
    e = pl.lit(False)
    if mini is not None:
        e = e | (pl.col(colonne) < mini)
    if maxi is not None:
        e = e | (pl.col(colonne) > maxi)
    return e.fill_null(False)


def op_verifier_plage(df: pl.DataFrame, colonne: str, action: str, min=None, max=None) -> Resultat:  # noqa: A002
    """Valeurs IMPOSSIBLES (hors des regles de bon sens). action : vider | signaler | supprimer.
    Pas de « plafonner » ici : ramener un age de 245 ans a 120 fabriquerait une fausse valeur."""
    _verifier_colonne(df, colonne)
    if df[colonne].dtype not in NUMERIQUES:
        raise ValueError(f"« {colonne} » n'est pas numérique : convertissez-la d'abord.")
    hors = _hors_plage(colonne, min, max)
    n = int(df.select(hors.sum()).item())
    plage = f"{'' if min is None else _n(min)} à {'' if max is None else _n(max)}".strip()
    if action == "vider":
        out = df.with_columns(pl.when(hors).then(None).otherwise(pl.col(colonne)).alias(colonne))
        msg = f"Valeurs impossibles de « {colonne} » vidées (hors {plage})"
    elif action == "signaler":
        out = df.with_columns(hors.alias(f"{colonne}_impossible"))
        msg = f"Valeurs impossibles de « {colonne} » signalées (colonne « {colonne}_impossible »)"
    elif action == "supprimer":
        out = df.filter(~hors)
        msg = f"Lignes avec une valeur impossible dans « {colonne} » supprimées"
    else:
        raise ValueError(f"Action inconnue : {action}")
    return Resultat(out, n, msg)


def op_dates_futures(df: pl.DataFrame, colonne: str, action: str) -> Resultat:
    """Une date de commande, d'inscription ou de naissance ne peut pas etre dans le futur."""
    _verifier_colonne(df, colonne)
    if df[colonne].dtype not in (pl.Date, pl.Datetime):
        raise ValueError(f"« {colonne} » n'est pas une date : convertissez-la d'abord.")
    futur = (pl.col(colonne).cast(pl.Date) > pl.lit(date.today())).fill_null(False)
    n = int(df.select(futur.sum()).item())
    if action == "vider":
        out, msg = df.with_columns(pl.when(futur).then(None).otherwise(pl.col(colonne)).alias(colonne)), \
            f"Dates futures de « {colonne} » vidées"
    elif action == "signaler":
        out, msg = df.with_columns(futur.alias(f"{colonne}_future")), f"Dates futures de « {colonne} » signalées"
    elif action == "supprimer":
        out, msg = df.filter(~futur), f"Lignes avec une date future dans « {colonne} » supprimées"
    else:
        raise ValueError(f"Action inconnue : {action}")
    return Resultat(out, n, msg)


# Catalogue : nom technique -> (fonction, libelle, types de colonne compatibles)
# « * » = toutes les colonnes ; None = l'operation ne cible pas une colonne precise.
OPERATIONS = {
    "supprimer_doublons": (op_supprimer_doublons, "Supprimer les doublons", None),
    "remplir_vides": (op_remplir_vides, "Remplacer les valeurs vides", "*"),
    "supprimer_lignes_vides": (op_supprimer_lignes_vides, "Supprimer les lignes vides", "*"),
    "traiter_aberrantes": (op_traiter_aberrantes, "Traiter les valeurs inhabituelles (statistique)", {"nombre"}),
    "verifier_plage": (op_verifier_plage, "Traiter les valeurs impossibles (hors d'une plage)", {"nombre"}),
    "dates_futures": (op_dates_futures, "Traiter les dates dans le futur", {"date"}),
    "nettoyer_texte": (op_nettoyer_texte, "Nettoyer le texte (espaces, majuscules)",
                       {"texte", "email", "telephone", "nombre (texte)", "date (texte)"}),
    "convertir_nombre": (op_convertir_nombre, "Convertir en nombre", {"nombre (texte)", "texte"}),
    "convertir_date": (op_convertir_date, "Convertir en date", {"date (texte)", "texte"}),
    "remplacer": (op_remplacer, "Remplacer une valeur", "*"),
    "renommer": (op_renommer, "Renommer une colonne", "*"),
    "supprimer_colonnes": (op_supprimer_colonnes, "Supprimer des colonnes", "*"),
}


def colonnes_compatibles(df: pl.DataFrame, op: str) -> list[str]:
    types = OPERATIONS[op][2]
    if types is None or types == "*":
        return list(df.columns)
    return [c for c in df.columns if type_colonne(df[c]) in types]


# =====================================================================================
# 4. Recette : appliquer, annuler, reutiliser
# =====================================================================================
@dataclass
class Etape:
    operation: dict
    n: int
    message: str


def appliquer(df: pl.DataFrame, recette: list[dict]) -> tuple[pl.DataFrame, list[Etape]]:
    """Rejoue toute la recette a partir du fichier d'origine. Une etape impossible (colonne absente
    dans le fichier du mois suivant, par exemple) arrete tout avec un message clair."""
    journal = []
    for i, operation in enumerate(recette, start=1):
        params = {k: v for k, v in operation.items() if k != "op"}
        if operation.get("op") not in OPERATIONS:
            raise ValueError(f"Étape {i} : opération inconnue « {operation.get('op')} »")
        try:
            r = OPERATIONS[operation["op"]][0](df, **params)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Étape {i} ({OPERATIONS[operation['op']][1]}) : {e}") from e
        df = r.df
        journal.append(Etape(operation, r.n, r.message))
    return df, journal


def recette_vers_json(recette: list[dict]) -> str:
    return json.dumps({"format": "recette-nettoyage", "version": 1, "operations": recette},
                      ensure_ascii=False, indent=2)


def recette_depuis_json(texte: str) -> list[dict]:
    d = json.loads(texte)
    if d.get("format") != "recette-nettoyage" or not isinstance(d.get("operations"), list):
        raise ValueError("Ce fichier n'est pas une recette de nettoyage.")
    return d["operations"]


# =====================================================================================
# 5. Propositions automatiques
# =====================================================================================
@dataclass
class Proposition:
    titre: str
    explication: str
    n: int
    choix: dict[str, dict]   # libelle du bouton -> operation a ajouter a la recette


FAUX_VIDES = {"N/A", "NA", "N.A.", "NULL", "NONE", "NAN", "-", "--", "?", "INCONNU", "NON RENSEIGNE",
              "NON RENSEIGNÉ", "#N/A", "VIDE"}


def faux_vides(serie: pl.Series) -> dict[str, int]:
    """Textes qui veulent dire « vide » (« N/A », « inconnu », « - »...) : invisibles pour un calcul de vides."""
    if serie.dtype != pl.String:
        return {}
    s = serie.drop_nulls().str.strip_chars()
    trouves = s.filter(s.str.to_uppercase().is_in(list(FAUX_VIDES)))
    return dict(trouves.value_counts().iter_rows()) if trouves.len() else {}


def proposer(df: pl.DataFrame, regles: dict[str, dict] | None = None,
             deja: set[tuple[str, str]] | None = None) -> list[Proposition]:
    """Ce que l'outil conseille. Il ne decide rien : il propose.
    L'ORDRE compte : valeurs impossibles, puis inhabituelles, AVANT de remplir les vides (sinon un age
    de 245 ans fausserait la moyenne utilisee pour remplacer les ages manquants).
    `deja` : (operation, colonne) deja dans la recette -> pas reproposees (sinon, apres un plafonnement,
    les bornes recalculees faisaient reapparaitre la meme proposition, sans fin)."""
    regles, deja = regles or {}, deja or set()
    profils, n_doublons = profiler(df)
    props: list[Proposition] = []
    texte, conversions, aberrantes, a_remplir, colonnes_vides = [], [], [], [], []
    if n_doublons:
        props.append(Proposition("Supprimer les doublons exacts", "Lignes strictement identiques sur toutes "
                                 "les colonnes : on garde la première.", n_doublons,
                                 {"Supprimer": {"op": "supprimer_doublons"}}))
    for p in profils:
        for valeur, n in faux_vides(df[p.nom]).items():
            texte.append(Proposition(f"« {valeur} » dans « {p.nom} » signifie probablement « vide »",
                                     "Tant que c'est du texte, ces valeurs ne sont pas comptées comme manquantes.",
                                     n, {"Remplacer par du vide": {"op": "remplacer", "colonne": p.nom,
                                                                    "rechercher": valeur, "remplacer": ""}}))
        if p.n_espaces:
            texte.append(Proposition(f"Espaces en trop dans « {p.nom} »", "Espaces au début, à la fin ou "
                                     "en double : invisibles, mais ils empêchent les comparaisons.", p.n_espaces,
                                     {"Nettoyer": {"op": "nettoyer_texte", "colonnes": [p.nom]}}))
        if p.type == "nombre (texte)":
            conversions.append(Proposition(f"« {p.nom} » contient des nombres écrits comme du texte",
                                     f"Exemples : {', '.join(p.exemples)}. Convertis, ils permettront les calculs "
                                     "(moyenne, médiane, valeurs aberrantes).", df.height - p.n_vides,
                                     {"Convertir": {"op": "convertir_nombre", "colonne": p.nom}}))
        if p.type == "date (texte)":
            conversions.append(Proposition(f"« {p.nom} » contient des dates écrites comme du texte",
                                     f"Exemples : {', '.join(p.exemples)}.", df.height - p.n_vides,
                                     {"Convertir": {"op": "convertir_date", "colonne": p.nom}}))
        if p.n_vides and p.type == "nombre":
            asym = abs(p.stats.get("asymetrie", 0))
            conseil = "mediane" if asym > 1 else "moyenne"
            pourquoi = (f"Quelques valeurs très éloignées tirent la moyenne ({p.stats['moyenne']:.2f}) : "
                        f"la médiane ({p.stats['mediane']:.2f}) est plus représentative."
                        if conseil == "mediane" else
                        f"Valeurs bien réparties : moyenne ({p.stats['moyenne']:.2f}) et médiane "
                        f"({p.stats['mediane']:.2f}) sont proches.")
            ordre = ["mediane", "moyenne"] if conseil == "mediane" else ["moyenne", "mediane"]
            a_remplir.append(Proposition(f"{p.pct_vides:.1%} de vides dans « {p.nom} »".replace(".", ",").replace("%", " %"),
                                     pourquoi + " Remplacer des vides fausse un peu les analyses : à décider "
                                     "selon l'usage.", p.n_vides,
                                     {("Médiane" if m == "mediane" else "Moyenne"):
                                      {"op": "remplir_vides", "colonne": p.nom, "methode": m} for m in ordre}))
        r = regles.get(p.nom)
        if p.type == "nombre" and r and ("verifier_plage", p.nom) not in deja:
            n_imp = int(df.select(_hors_plage(p.nom, r["min"], r["max"]).sum()).item())
            if n_imp:
                plage = (f"entre {_n(r['min'])} et {_n(r['max'])}" if r["min"] is not None and r["max"] is not None
                         else f"au moins {_n(r['min'])}" if r["min"] is not None else f"au plus {_n(r['max'])}")
                impossibles = df.filter(_hors_plage(p.nom, r["min"], r["max"]))[p.nom].unique().sort().head(5)
                aberrantes.append(Proposition(
                    f"Valeurs impossibles dans « {p.nom} »",
                    f"{', '.join(_n(v) for v in impossibles.to_list())} : hors de la plage possible ({plage} ; "
                    f"{r['raison']}, source : {r['source']}). Conseil : « Vider ». La vraie valeur est "
                    "inconnue, mieux vaut une case vide qu'une valeur inventée.", n_imp,
                    {"Vider": {"op": "verifier_plage", "colonne": p.nom, "action": "vider",
                                           "min": r["min"], "max": r["max"]},
                     "Signaler": {"op": "verifier_plage", "colonne": p.nom, "action": "signaler",
                                  "min": r["min"], "max": r["max"]},
                     "Supprimer": {"op": "verifier_plage", "colonne": p.nom, "action": "supprimer",
                                              "min": r["min"], "max": r["max"]}}))
        if p.type == "nombre" and ("traiter_aberrantes", p.nom) not in deja:
            # Statistique calculee sur les seules valeurs POSSIBLES ; tolerance elargie (k = 3) pour les
            # colonnes tres asymetriques (montants : quelques gros achats sont normaux).
            ok = df[p.nom].drop_nulls()
            if r:
                ok = df.filter(~_hors_plage(p.nom, r["min"], r["max"]))[p.nom].drop_nulls()
            asym = abs(float(ok.skew())) if ok.len() > 2 and ok.std() else 0.0
            k = 3.0 if asym > 1 else 1.5
            b = bornes_iqr(ok, k)
            if b:
                inhab = ok.filter((ok < b[0]) | (ok > b[1]))
                bas = max(b[0], r["min"]) if r and r["min"] is not None else b[0]   # pas de -5 EUR affiche
                haut = min(b[1], r["max"]) if r and r["max"] is not None else b[1]
                if inhab.len():
                    exemples = ", ".join(_n(v) for v in inhab.unique().sort().head(5).to_list())
                    aberrantes.append(Proposition(
                        f"Valeurs inhabituelles dans « {p.nom} »",
                        f"{exemples} : très éloignées des autres valeurs (habituellement entre {_n(round(bas, 2))} et "
                        f"{_n(round(haut, 2))}). Ce sont peut-être de vraies valeurs : conseil, « Signaler » et "
                        "vérifier avant de modifier.",
                        inhab.len(),
                        {"Signaler": {"op": "traiter_aberrantes", "colonne": p.nom, "action": "signaler",
                                                  "k": k},
                         "Vider": {"op": "traiter_aberrantes", "colonne": p.nom, "action": "vider", "k": k},
                         "Plafonner": {"op": "traiter_aberrantes", "colonne": p.nom, "action": "plafonner", "k": k},
                         "Supprimer": {"op": "traiter_aberrantes", "colonne": p.nom, "action": "supprimer", "k": k}}))
        if p.type == "date" and ("dates_futures", p.nom) not in deja:
            n_fut = int(df.select((pl.col(p.nom).cast(pl.Date) > pl.lit(date.today())).fill_null(False).sum()).item())
            if n_fut:
                aberrantes.append(Proposition(
                    f"Dates dans le futur dans « {p.nom} »",
                    "Une commande ou une inscription ne peut pas avoir lieu dans le futur : erreur de saisie probable.",
                    n_fut, {"Vider": {"op": "dates_futures", "colonne": p.nom, "action": "vider"},
                            "Signaler": {"op": "dates_futures", "colonne": p.nom, "action": "signaler"},
                            "Supprimer": {"op": "dates_futures", "colonne": p.nom, "action": "supprimer"}}))
        if p.n_vides and p.type not in ("nombre",) and p.pct_vides > 0.5:
            colonnes_vides.append(Proposition(f"« {p.nom} » est vide à {p.pct_vides:.0%}",
                                     "Une colonne presque vide apporte peu d'information.", p.n_vides,
                                     {"Supprimer la colonne": {"op": "supprimer_colonnes", "colonnes": [p.nom]}}))
    return props + texte + conversions + aberrantes + a_remplir + colonnes_vides




# =====================================================================================
# 6. Fichier d'exemple (pour essayer l'atelier sans avoir de fichier)
# =====================================================================================
def exemple_ventes(n: int = 5_000, seed: int = 7) -> pl.DataFrame:
    """Un export de ventes realiste et sale : montants au format francais, dates en plusieurs formats,
    quantites et ages aberrants, vides, « N/A », espaces en trop, lignes en double."""
    import random

    from faker import Faker

    rng, fake = random.Random(seed), Faker("fr_FR")
    Faker.seed(seed)
    villes = ["Paris", "Lyon", "Marseille", "Lille", "Nantes", "Bordeaux", "Toulouse", "Rennes"]
    lignes = []
    for i in range(1, n + 1):
        montant = round(rng.lognormvariate(3.8, 0.7), 2)            # asymetrique : beaucoup de petits paniers
        d = fake.date_between(start_date=date(2025, 1, 1), end_date=date(2025, 12, 31))
        lignes.append({
            "id_commande": f"CMD-{i:06d}",
            "client": (" " if rng.random() < 0.06 else "") + fake.name() + ("  " if rng.random() < 0.05 else ""),
            "email": fake.email() if rng.random() > 0.05 else None,
            "ville": rng.choice(villes) if rng.random() > 0.04 else rng.choice(["", "N/A"]),
            "canal": rng.choice(["web", "magasin", "téléphone"]),
            "montant": (f"{montant:,.2f} €".replace(",", " ").replace(".", ",") if rng.random() > 0.03 else None),
            "quantite": rng.randint(1, 6) if rng.random() > 0.01 else rng.choice([0, 250, 999]),
            "age_client": (rng.randint(18, 80) if rng.random() > 0.06 else None) if rng.random() > 0.01
            else rng.choice([2, 150, 212]),
            "date_commande": d.strftime(rng.choice(["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"])),
            "statut": rng.choice(["payé"] * 8 + ["remboursé", "N/A"]),
        })
    df = pl.DataFrame(lignes)
    doublons = df.sample(int(n * 0.03), seed=seed)              # exports fusionnes deux fois
    return pl.concat([df, doublons]).sample(fraction=1.0, shuffle=True, seed=seed)
