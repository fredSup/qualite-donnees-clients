import json

import polars as pl

from qdc import ia


def _silver():
    df = pl.DataFrame({c: ["x", "y", "z"] for c in ia.CHAMPS})
    return df.with_columns(record_id=pl.Series(["A", "B", "C"]), nom=pl.Series(["Martin", "Martin", "Durand"]))


def _bonne_reponse(messages, meme=True, confiance=0.9):
    lot = json.loads(messages[1]["content"])["paires"]
    return json.dumps({"decisions": [{"id": p["id"], "nom_fiche_1": p["fiche_1"]["nom"], "meme_personne": meme,
                                      "confiance": confiance, "raison": "test"} for p in lot]})


def test_lot_puis_cache(tmp_path):
    """Les paires deja jugees ne sont jamais renvoyees au LLM."""
    appels = []

    def faux_llm(messages):
        appels.append(len(json.loads(messages[1]["content"])["paires"]))
        return _bonne_reponse(messages)

    paires = pl.DataFrame({"id_a": ["A", "A"], "id_b": ["B", "C"]})
    base = tmp_path / "t.duckdb"
    ia.traiter(paires, _silver(), base, appeler=faux_llm, pause=0)
    d = ia.traiter(paires, _silver(), base, appeler=faux_llm, pause=0)
    assert appels == [2]  # un seul appel : la 2e fois, tout vient du cache
    assert d.height == 2


def test_cache_suit_le_contenu_et_non_les_identifiants(tmp_path):
    """Memes identifiants, contenu different (donnees regenerees) => nouvelle question."""
    appels = []

    def faux_llm(messages):
        appels.append(1)
        return _bonne_reponse(messages)

    paires = pl.DataFrame({"id_a": ["A"], "id_b": ["B"]})
    base = tmp_path / "t.duckdb"
    ia.traiter(paires, _silver(), base, appeler=faux_llm, pause=0)
    ia.traiter(paires, _silver().with_columns(prenom=pl.lit("autre")), base, appeler=faux_llm, pause=0)
    assert len(appels) == 2


def test_reponse_invalide_puis_nouvelle_tentative(monkeypatch):
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    lot = [{"id": "A|B", "fiche_1": {"nom": "Martin"}, "fiche_2": {}}]
    reponses = iter(["pas du json",
                     json.dumps({"decisions": ["texte au lieu d'objet",
                                               {"id": "A|B", "nom_fiche_1": "MARTIN", "meme_personne": False}]})])
    # JSON illisible -> on relance ; puis les elements mal formes sont ignores, le reste garde
    assert ia.juger(lot, lambda m: next(reponses))[0]["meme_personne"] is False


def test_reponse_sur_la_mauvaise_paire_rejetee(monkeypatch):
    """Le LLM recopie un autre nom que celui de la fiche 1 : il confond les paires -> on jette."""
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    lot = [{"id": "A|B", "fiche_1": {"nom": "Martin"}, "fiche_2": {}}]
    rep = json.dumps({"decisions": [{"id": "A|B", "nom_fiche_1": "Lefebvre", "meme_personne": True}]})
    assert ia.juger(lot, lambda m: rep, essais=1) == []


def test_les_ids_inventes_par_le_llm_sont_ignores():
    lot = [{"id": "A|B", "fiche_1": {"nom": None}, "fiche_2": {}}]
    rep = json.dumps({"decisions": [{"id": "Z|Z", "meme_personne": True}, {"id": "A|B", "meme_personne": True}]})
    assert [d["id"] for d in ia.juger(lot, lambda m: rep)] == ["A|B"]


def test_comparaison_champ_par_champ():
    assert ia.comparer("Dupont", "DUPONT") == "identique"
    assert ia.comparer("Dupont", "Dupond") == "proche"
    assert ia.comparer("Dupont", "Lefebvre") == "different"
    assert ia.comparer(None, "x") == "inconnu"


def test_evaluation_ia():
    verite = pl.DataFrame({"record_id": ["A", "B", "C"], "person_id": [1, 1, 2]})
    decisions = pl.DataFrame({"id": ["A|B", "A|C"], "meme_personne": [True, True],
                              "confiance": [0.9, 0.95], "raison": ["", ""]})
    ev = ia.evaluer_ia(decisions, verite)
    assert ev["precision_fusions"] == 0.5 and ev["decidees_seules"] == 2


def test_nom_recopie_tolerant():
    assert ia._meme_nom("T. Ménard", "Menard") and ia._meme_nom("MENARD", "Ménard")
    assert ia._meme_nom("Menarrd", "Menard")       # faute de frappe : meme personne de nom
    assert not ia._meme_nom("Lefebvre", "Menard")  # autre paire


def test_reponse_valide_partielle_pas_relancee(monkeypatch):
    """Une reponse valide n'est JAMAIS relancee, meme si une decision est rejetee (sinon on paie 4 fois)."""
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    appels = []
    lot = [{"id": "A|B", "fiche_1": {"nom": "Martin"}, "fiche_2": {}},
           {"id": "A|C", "fiche_1": {"nom": "Durand"}, "fiche_2": {}}]

    def llm(m):
        appels.append(1)
        return json.dumps({"decisions": [{"id": "A|B", "nom_fiche_1": "Martin", "meme_personne": True},
                                         {"id": "A|C", "nom_fiche_1": "Lefebvre", "meme_personne": True}]})

    assert [d["id"] for d in ia.juger(lot, llm)] == ["A|B"] and len(appels) == 1


def test_attente_lue_dans_le_message_groq():
    assert ia._attente_conseillee("Please try again in 7.5s.", 99) == 8.5
    assert ia._attente_conseillee("Please try again in 1m2s.", 99) == 63
    assert ia._attente_conseillee("autre", 99) == 99


def test_quota_journalier_arret_propre(tmp_path):
    def llm(m):
        raise RuntimeError("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    paires = pl.DataFrame({"id_a": ["A"], "id_b": ["B"]})
    d = ia.traiter(paires, _silver(), tmp_path / "t.duckdb", appeler=llm, pause=0)
    assert d.is_empty()


def test_verrou_dates_de_naissance():
    """Meme si l'IA dit « meme personne », des dates de naissance differentes l'interdisent."""
    q = {"id": "A|B", "fiche_1": {"nom": "Martin"}, "fiche_2": {}, "comparaison": {"date_naissance": "different"}}
    rep = json.dumps({"decisions": [{"id": "A|B", "nom_fiche_1": "Martin", "meme_personne": True, "confiance": 0.95}]})
    d = ia.juger([q], lambda m: rep)[0]
    assert d["meme_personne"] is False and d["raison"].startswith("Verrou")


def test_inversion_prenom_nom_detectee():
    s = pl.DataFrame({c: ["", ""] for c in ia.CHAMPS}).with_columns(
        record_id=pl.Series(["A", "B"]), prenom=pl.Series(["Julien", "Patrick"]), nom=pl.Series(["Patrick", "Julien"]))
    q = ia.preparer_lot(pl.DataFrame({"id_a": ["A"], "id_b": ["B"]}), s)[0]
    assert q["comparaison"]["prenom_et_nom_inverses"] == "oui"
