"""Test de bout en bout : tout le pipeline sur un petit jeu de donnees, jusqu'au rapport."""
import duckdb
import polars as pl
import pytest

from qdc import dedoublonnage, fusion, ingestion, nettoyage, rapport
from qdc.generation import Config, generer


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline")
    generer(Config(n_personnes=400, seed=5), d / "raw")
    db = d / "qdc.duckdb"
    bronze = ingestion.ingerer(d / "raw", db)
    silver = nettoyage.nettoyer(bronze)
    paires, groupes = dedoublonnage.executer(silver)
    gold = fusion.fusionner(silver, groupes)
    with duckdb.connect(str(db)) as con:
        for nom, df in [("silver_clients", silver), ("gold_clients", gold),
                        ("paires", paires.select("id_a", "id_b", "score", "decision"))]:
            con.register("t", df.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {nom} AS SELECT * FROM t")
    return db


def test_pipeline_complet_et_faits_coherents(base):
    f = rapport.collecter_faits(base)
    fiches = int(f["nombre_de_fiches_avant_traitement"].replace(" ", ""))
    clients = int(f["nombre_de_clients_distincts_apres_traitement"].replace(" ", ""))
    assert 400 <= clients < fiches  # on a bien regroupe des doublons, sans descendre sous 400 personnes
    assert float(f["score_qualite_apres_traitement"].replace(",", ".")) > float(f["score_qualite_avant_traitement"].replace(",", "."))


def test_rapport_sans_ia_ne_contient_que_des_chiffres_verifies(base):
    f = rapport.collecter_faits(base)
    corps, auteur, _ = rapport.rediger(f, appeler=lambda m: (_ for _ in ()).throw(RuntimeError("pas d'API")))
    assert auteur.startswith("modele fixe")
    assert rapport.verifier_chiffres(corps, f) == []


def test_chiffre_invente_par_l_ia_detecte(base):
    f = rapport.collecter_faits(base)
    texte = f"## Synthese\nScore de {f['score_qualite_avant_traitement']} a {f['score_qualite_apres_traitement']} / 100, soit 12 500 euros economises."
    corps, auteur, suspects = rapport.rediger(f, appeler=lambda m: texte)
    assert auteur == "IA" and suspects == ["12 500"]
    assert "Chiffres non retrouves" in rapport.assembler(f, corps, auteur, suspects)


def test_texte_avant_le_premier_titre_retire():
    f = {"x": "1"}
    corps, _, _ = rapport.rediger(f, appeler=lambda m: "Voici le rapport :\n## Synthese\nTexte.")
    assert corps.startswith("## Synthese")


def test_rapport_compte_les_cas_vus_par_l_ia(base):
    """Seules les reponses du cache qui correspondent aux paires ACTUELLES sont comptees.
    (Couvre aussi un bug DuckDB 2 : fetchone() n'existe plus sur une relation.)"""
    from qdc import ia

    with duckdb.connect(str(base)) as con:
        silver = con.sql("SELECT * FROM silver_clients").pl()
        ambigues = con.sql("SELECT id_a, id_b FROM paires WHERE decision = 'ambigu' LIMIT 1").pl()
    assert ambigues.height == 1
    vraie = ia.preparer_lot(ambigues, silver)[0]
    ia.enregistrer(base, [{"id": "x", "meme_personne": True, "confiance": 0.9},
                          {"id": "y", "meme_personne": False, "confiance": 0.9}],
                   {"x": vraie["_cle"], "y": "ancienne-reponse-d-un-autre-prompt"})
    assert rapport.collecter_faits(base)["cas_douteux_deja_examines_par_ia"] == "1"


def test_espaces_insecables_acceptees():
    """L'IA ecrit parfois « 18 170 » avec une espace insecable fine : ce n'est pas un chiffre suspect."""
    assert rapport.verifier_chiffres("18 170 fiches et 7 778 doublons", {"a": "18 170", "b": "7 778"}) == []


def test_faits_detailles_par_source(base):
    f = rapport.collecter_faits(base)
    total = sum(int(d["corrections_appliquees"].get("telephone: format E.164", "0").replace(" ", ""))
                for d in f["sources"].values())
    assert str(total) == f["corrections_appliquees_toutes_sources"]["telephone: format E.164"].replace(" ", "")
