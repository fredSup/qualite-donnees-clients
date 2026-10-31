from datetime import date

import polars as pl

from qdc.dedoublonnage import evaluer, paires_candidates, preparer, regrouper, scorer


def _silver(rows):
    base = {"record_id": None, "source": "crm", "civilite": None, "prenom": None, "nom": None, "email": None,
            "telephone": None, "adresse": None, "code_postal": None, "ville": None, "date_naissance": None,
            "date_creation": None, "date_dernier_contact": None, "corrections": [], "anomalies": []}
    texte = {c: pl.String for c in base if c not in ("date_naissance", "date_creation", "date_dernier_contact",
                                                       "corrections", "anomalies")}
    return pl.DataFrame([base | r for r in rows], schema_overrides=texte | {
        "date_naissance": pl.Date, "date_creation": pl.Date, "date_dernier_contact": pl.Date,
        "corrections": pl.List(pl.String), "anomalies": pl.List(pl.String)})


def _decisions(rows):
    p = preparer(_silver(rows))
    res = scorer(paires_candidates(p), p)
    return {(r["id_a"], r["id_b"]): r["decision"] for r in res.iter_rows(named=True)}


def test_doublon_avec_faute_de_frappe_et_format_different():
    d = _decisions([
        {"record_id": "A", "prenom": "Jean", "nom": "Dupont", "code_postal": "75011",
         "telephone": "+33612345678", "date_naissance": date(1980, 5, 2)},
        {"record_id": "B", "prenom": "Jean", "nom": "Dupond", "code_postal": "75011", "source": "boutique",
         "telephone": "+33612345678", "date_naissance": date(1980, 5, 2)},
    ])
    assert d[("A", "B")] == "doublon"


def test_couple_partageant_le_telephone_n_est_pas_fusionne():
    """Le piege : meme nom, meme adresse, meme fixe... mais deux personnes."""
    d = _decisions([
        {"record_id": "A", "civilite": "Mme", "prenom": "Claire", "nom": "Martin", "telephone": "+33145678901",
         "adresse": "12 rue des Lilas", "code_postal": "75011"},
        {"record_id": "B", "civilite": "M.", "prenom": "Paul", "nom": "Martin", "telephone": "+33145678901",
         "adresse": "12 rue des Lilas", "code_postal": "75011"},
    ])
    assert d[("A", "B")] == "different"


def test_regroupement_transitif():
    g = regrouper(pl.Series(["A", "B", "C", "D"]), pl.DataFrame({"id_a": ["A", "B"], "id_b": ["B", "C"]}))
    c = dict(g.iter_rows())
    assert c["A"] == c["B"] == c["C"] != c["D"]


def test_evaluation_precision_rappel():
    verite = pl.DataFrame({"record_id": ["A", "B", "C", "D"], "person_id": [1, 1, 2, 2]})
    # On a fusionne A+B (juste) et rate C+D
    groupes = pl.DataFrame({"record_id": ["A", "B", "C", "D"], "client": ["A", "A", "C", "D"]})
    e = evaluer(groupes, verite)
    assert e.precision == 1.0 and e.rappel == 0.5
