import polars as pl

from qdc import pipeline
from qdc.generation import Config, generer


def test_pipeline_et_decisions_humaines(tmp_path):
    generer(Config(n_personnes=300, seed=9), tmp_path / "raw")
    base = tmp_path / "qdc.duckdb"
    r = pipeline.executer_tout(tmp_path / "raw", base)
    assert pipeline.pret(base) and 300 <= r["clients"] < r["fiches"]

    # Un humain fusionne une paire ambigue : le referentiel perd un client
    amb = pipeline.lire(base, "paires").filter(pl.col("decision") == "ambigu").row(0, named=True)
    avant = r["clients"]
    pipeline.enregistrer_decision(base, amb["id_a"], amb["id_b"], "fusionner")
    assert pipeline.construire_referentiel(base).height == avant - 1

    # ... puis change d'avis : la fusion est annulee
    pipeline.enregistrer_decision(base, amb["id_a"], amb["id_b"], "separer")
    assert pipeline.construire_referentiel(base).height == avant
