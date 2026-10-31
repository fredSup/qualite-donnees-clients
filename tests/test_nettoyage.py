from datetime import date

import polars as pl

from qdc.ingestion import SCHEMA_COMMUN
from qdc.nettoyage import journal, nettoyer


def _bronze(**lignes):
    """Construit une petite table bronze : chaque argument est une ligne (dict)."""
    rows = []
    for rid, champs in lignes.items():
        row = {c: None for c in SCHEMA_COMMUN} | {"record_id": rid, "source": "boutique"} | champs
        rows.append(row)
    return pl.DataFrame(rows, schema={c: pl.String for c in SCHEMA_COMMUN})


def _ligne(silver, rid):
    return silver.filter(pl.col("record_id") == rid).row(0, named=True)


def test_decoupe_nom_complet_dans_les_deux_ordres():
    s = nettoyer(_bronze(a={"nom_complet": "Nathalie TOUSSAINT"}, b={"nom_complet": "TOUSSAINT Nathalie"},
                         c={"nom_complet": "M. Petit", "prenom": "Gérard"}))
    for rid in ("a", "b"):
        assert (_ligne(s, rid)["prenom"], _ligne(s, rid)["nom"]) == ("Nathalie", "Toussaint")
    c = _ligne(s, "c")
    assert (c["civilite"], c["prenom"], c["nom"]) == ("M.", "Gérard", "Petit")


def test_prenom_sans_accent_reconnu():
    s = nettoyer(_bronze(a={"nom_complet": "Joseph Etienne"}, b={"nom_complet": "Durand Etienne"}))
    assert _ligne(s, "b")["prenom"] == "Etienne"  # Etienne = Étienne


def test_on_corrige_le_sur_et_on_signale_le_reste():
    s = nettoyer(_bronze(a={"email": " Jean.Dupont@Gmail.com ", "telephone": "06 12 34 56 78",
                            "code_postal": "35000", "ville": "REIMS", "nom": "dupont"},
                         b={"email": "jean.dupontgmail.com", "date_naissance": "1867-03-12", "nom": "x"}),
                 aujourd_hui=date(2026, 10, 31))
    a, b = _ligne(s, "a"), _ligne(s, "b")
    assert a["email"] == "jean.dupont@gmail.com" and a["telephone"] == "+33612345678"
    assert a["ville"] == "Reims"  # on ne "corrige" PAS la ville...
    assert "code postal incoherent avec la ville" in a["anomalies"]  # ... on la signale
    assert b["email"] is None and "email invalide (supprime)" in b["anomalies"]
    assert b["date_naissance"] is None


def test_enrichissements():
    s = nettoyer(_bronze(a={"nom_complet": "Camille Martin", "code_postal": "6000"}))
    a = _ligne(s, "a")
    assert a["code_postal"] == "06000" and a["ville"] == "Nice"
    assert "ville: deduite du code postal" in a["corrections"]


def test_journal_compte_les_operations():
    j = journal(nettoyer(_bronze(a={"telephone": "0612345678", "nom": "a"}, b={"telephone": "0712345678", "nom": "b"})))
    assert j.filter(pl.col("operation") == "telephone: format E.164")["n"].item() == 2
