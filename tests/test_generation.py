import polars as pl

from qdc.generation import Config, generer


def _lire(dossier):
    return (
        pl.read_csv(dossier / "raw" / "crm_clients.csv", separator=";", infer_schema_length=0),
        pl.read_parquet(dossier / "raw" / "support_clients.parquet"),
        pl.read_csv(dossier / "verite" / "verite.csv"),
    )


def test_reproductible(tmp_path):
    """Meme graine => exactement les memes donnees (indispensable pour comparer deux versions de l'outil)."""
    generer(Config(n_personnes=300, seed=7), tmp_path / "a" / "raw")
    generer(Config(n_personnes=300, seed=7), tmp_path / "b" / "raw")
    for x, y in zip(_lire(tmp_path / "a"), _lire(tmp_path / "b")):
        assert x.equals(y)


def test_verite_complete(tmp_path):
    """Chaque enregistrement produit est present une et une seule fois dans la verite."""
    generer(Config(n_personnes=300, seed=1), tmp_path / "raw")
    crm, support, verite = _lire(tmp_path)
    assert verite["record_id"].is_unique().all()
    ids = set(verite["record_id"])
    assert set(crm["id_crm"]) <= ids and set(support["ticket_client_id"]) <= ids
    assert verite["person_id"].n_unique() == 300


def test_donnees_vraiment_sales(tmp_path):
    """Le jeu doit contenir des defauts, sinon il ne teste rien."""
    generer(Config(n_personnes=1000, seed=3), tmp_path / "raw")
    crm, _, verite = _lire(tmp_path)
    assert crm["Email"].null_count() > 0
    assert crm["Tél"].str.contains(r"\+33").any() and crm["Tél"].str.contains(r"\.").any()
    assert verite.height > verite["person_id"].n_unique()  # il y a des doublons
