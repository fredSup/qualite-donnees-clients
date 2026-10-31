from datetime import date

import polars as pl

from qdc.fusion import fusionner

COLS = {"record_id": pl.String, "source": pl.String, "civilite": pl.String, "prenom": pl.String,
        "nom": pl.String, "email": pl.String, "telephone": pl.String, "adresse": pl.String,
        "code_postal": pl.String, "ville": pl.String, "date_naissance": pl.Date, "date_creation": pl.Date,
        "date_dernier_contact": pl.Date, "corrections": pl.List(pl.String), "anomalies": pl.List(pl.String)}


def _silver(rows):
    base = {c: None for c in COLS} | {"corrections": [], "anomalies": []}
    return pl.DataFrame([base | r for r in rows], schema=COLS)


def _client():
    silver = _silver([
        {"record_id": "A", "source": "crm", "prenom": "A.", "nom": "Dupond", "email": "a.dupont@pro.fr",
         "telephone": "+33145678901", "adresse": "1 rue Ancienne", "code_postal": "69003", "ville": "Lyon",
         "date_creation": date(2019, 1, 1), "date_naissance": date(1980, 5, 2)},
        {"record_id": "B", "source": "boutique", "prenom": "Aurélie", "nom": "Dupont", "email": "aurelie@gmail.com",
         "telephone": "+33612345678", "adresse": "8 rue Neuve", "code_postal": "75011", "ville": "Paris",
         "date_creation": date(2025, 6, 1), "date_naissance": date(1980, 5, 2)},
        {"record_id": "C", "source": "support", "nom": "Dupont", "code_postal": "75011", "ville": "Paris",
         "date_dernier_contact": date(2026, 9, 1)},
    ])
    groupes = pl.DataFrame({"record_id": ["A", "B", "C"], "client": ["A", "A", "A"]})
    return fusionner(silver, groupes).row(0, named=True)


def test_une_seule_fiche_avec_les_meilleures_valeurs():
    c = _client()
    assert c["n_fiches"] == 3
    assert c["prenom"] == "Aurélie"            # le plus complet, pas l'initiale
    assert c["nom"] == "Dupont"                # la majorite (2 contre 1) gagne sur la faute de frappe
    assert c["telephone"] == "+33612345678"    # mobile prefere au fixe
    assert sorted(c["emails"]) == ["a.dupont@pro.fr", "aurelie@gmail.com"]  # on garde tous les e-mails


def test_adresse_bloc_coherent_et_recent():
    c = _client()
    # La fiche la plus recente AVEC une adresse est B : rue, CP et ville viennent tous de B
    assert (c["adresse"], c["code_postal"], c["ville"]) == ("8 rue Neuve", "75011", "Paris")
    assert c["lignage"]["origine_code_postal"] == "B"


def test_conflit_de_date_de_naissance_signale():
    silver = _silver([
        {"record_id": "A", "source": "crm", "prenom": "Claire", "nom": "Martin", "date_naissance": date(1980, 1, 1)},
        {"record_id": "B", "source": "crm", "prenom": "Claire", "nom": "Martin", "date_naissance": date(1955, 7, 7)},
    ])
    c = fusionner(silver, pl.DataFrame({"record_id": ["A", "B"], "client": ["A", "A"]})).row(0, named=True)
    assert "dates de naissance differentes" in c["conflits"]


def test_identifiant_client_stable():
    assert _client()["client_id"] == _client()["client_id"]


def test_prenom_majoritaire_plutot_que_faute_de_frappe():
    """« Luccie » est plus long que « Lucie », mais minoritaire : c'est une faute de frappe."""
    silver = _silver([
        {"record_id": "A", "source": "crm", "prenom": "Luccie", "nom": "Pons"},
        {"record_id": "B", "source": "boutique", "prenom": "Lucie", "nom": "Pons"},
        {"record_id": "C", "source": "support", "prenom": "Lucie", "nom": "Pons"},
        {"record_id": "D", "source": "crm", "prenom": "L.", "nom": "Pons"},
    ])
    c = fusionner(silver, pl.DataFrame({"record_id": list("ABCD"), "client": ["A"] * 4})).row(0, named=True)
    assert c["prenom"] == "Lucie"
