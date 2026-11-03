"""Le scenario de Frederic : un CRM et un fichier compta, avec des colonnes differentes."""
import polars as pl
import pytest

from qdc import ingestion, pipeline


def _fichiers():
    crm = pl.DataFrame({
        "id": ["C1", "C2", "C3"],
        "Prénom": ["Frédéric", "Claire", "Paul"],
        "Nom": ["Math", "Durand", "Martin"],
        "Mail": ["fredericmath@gmail.com", "claire.durand@free.fr", "paul.martin@orange.fr"],
        "Tél": ["06 12 34 56 78", "0698765432", "07 11 22 33 44"],
        "Date de naissance": ["12/03/1990", "1985-07-01", None],
    })
    compta = pl.DataFrame({
        "N° client": ["K-77", "K-78"],
        "Nom complet": ["MATH Frédéric", "Lefebvre Anne"],
        "Courriel": [" frederic.Math@gmail.com ", "anne.lefebvre@gmail.com"],
        "Téléphone portable": ["+33612345678", "+33600000001"],
        "IBAN": ["FR76 3000 6000 0112 3456 7890 189", "FR76 1111"],
        "Montant dû": ["120,50", "0"],
    })
    return [("CRM", crm, ingestion.deviner_correspondance(crm)),
            ("Compta", compta, ingestion.deviner_correspondance(compta))]


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "qdc.duckdb"
    pipeline.executer_fichiers(_fichiers(), b)
    return b


def test_correspondance_devinee():
    (_, _, crm), (_, _, compta) = _fichiers()
    assert crm["Mail"] == "email" and crm["Tél"] == "telephone" and crm["id"] == ingestion.CONSERVER
    assert compta["Nom complet"] == "nom_complet" and compta["IBAN"] == ingestion.CONSERVER


def test_frederic_reuni_en_une_seule_fiche(base):
    gold = pipeline.lire(base, "gold_clients")
    assert gold.height == 4  # Frederic, Claire, Paul, Anne
    f = gold.filter(pl.col("nom") == "Math").row(0, named=True)
    assert f["n_fiches"] == 2 and f["prenom"] == "Frédéric"
    assert sorted(f["emails"]) == ["frederic.math@gmail.com", "fredericmath@gmail.com"]  # on garde les deux


def test_colonnes_supplementaires_conservees(base):
    gold = pipeline.lire(base, "gold_clients")
    f = gold.filter(pl.col("nom") == "Math").row(0, named=True)
    assert f["IBAN"].startswith("FR76 3000") and f["Montant dû"] == "120,50"
    assert gold.filter(pl.col("nom") == "Durand")["IBAN"].item() is None  # Claire n'est pas en compta


def test_chaque_fichier_corrige_avec_le_meme_client_id(base):
    crm, compta = pipeline.fichier_corrige(base, "CRM"), pipeline.fichier_corrige(base, "Compta")
    assert crm.columns[:2] == ["client_id", "record_id"]
    assert crm.columns[2:8] == ["id", "Prénom", "Nom", "Mail", "Tél", "Date de naissance"]  # ordre d'origine
    id_crm = crm.filter(pl.col("Nom") == "Math")["client_id"].item()
    id_compta = compta.filter(pl.col("IBAN").str.starts_with("FR76 3000"))["client_id"].item()
    assert id_crm == id_compta  # le meme Frederic dans les deux services
    assert crm.filter(pl.col("Nom") == "Math")["Tél"].item() == "+33612345678"  # valeur corrigee
    assert compta["Courriel"][0] == "frederic.math@gmail.com"


def test_limites():
    df = pl.DataFrame({"email": ["a@b.fr"]})
    with pytest.raises(ValueError, match="entre 1 et"):
        ingestion.ingerer_fichiers([(f"S{i}", df, {"email": "email"}) for i in range(11)], None)
    with pytest.raises(ValueError, match="nom de source différent"):
        ingestion.ingerer_fichiers([("A", df, {"email": "email"}), ("A", df, {"email": "email"})], None)
    with pytest.raises(ValueError, match="au moins un nom"):
        ingestion.ingerer_fichiers([("A", pl.DataFrame({"x": [1]}), {"x": ingestion.CONSERVER})], None)


def test_cas_de_frederic_noms_proches_sans_email_ni_telephone(tmp_path):
    """Bugs trouves par Frederic : identifiants en double (meme prefixe), score « nan »,
    completude faussee par des champs qu'aucun fichier ne contient."""
    import math

    from qdc import diagnostic, fusion

    v1 = pl.DataFrame({"ID Client": [101, 102, 103], "Nom Client": ["Jean Dupont", "Marie Curie", "Pierre Martin"],
                       "Age": [34, 29, 250], "Statut Compte": ["Actif", "Inactif", "Actif"]})
    v2 = v1.with_columns(pl.Series("Age", [34, 29, 45]))
    base = tmp_path / "q.duckdb"
    fichiers = [(n, df, ingestion.deviner_correspondance(df))
                for n, df in (("jeu_de_donnees_test", v1), ("jeu_de_donnees_test_propre", v2))]
    pipeline.executer_fichiers(fichiers, base)
    bronze = pipeline.lire(base, "bronze_clients")
    assert bronze["record_id"].is_unique().all()
    for src in pipeline.sources(base):  # plus d'erreur « got 2 values »
        assert pipeline.fichier_corrige(base, src).height == 3
    gold = pipeline.lire(base, "gold_clients")
    df_g, app = fusion.pour_diagnostic(gold, ingestion.champs_collectes(base))
    d = diagnostic.diagnostiquer(df_g, applicable=app)
    assert not math.isnan(d.score_global)
    assert dict(d.resume.select("dimension", "taux").iter_rows())["Complétude"] == 1.0


def test_fichiers_de_frederic_identifiant_commun(tmp_path):
    """Les deux fichiers de test de Frederic partagent la colonne « ID Client » : elle est detectee
    comme identifiant commun, et chaque client n'apparait plus qu'une fois."""
    from pathlib import Path

    from qdc import atelier

    dossier = Path(__file__).parent / "donnees"
    noms = ["jeu_de_donnees_test_nettoyage_propre.csv", "jeu_de_donnees_test_nettoyage.xlsx"]
    tableaux = [atelier.charger((dossier / n).read_bytes(), n) for n in noms]
    communs = ingestion.identifiants_communs(tableaux)
    assert communs == {"ID Client"}
    fichiers = []
    for n, df in zip(noms, tableaux):
        c = ingestion.deviner_correspondance(df)
        c.update({col: "id_client" for col in df.columns if col in communs})
        fichiers.append((n.split(".")[0], df, c))
    base = tmp_path / "q.duckdb"
    r = pipeline.executer_fichiers(fichiers, base)
    ids = set().union(*[set(df["ID Client"].cast(pl.String).to_list()) for df in tableaux])
    assert r["clients"] == len(ids)


def _base_frederic(tmp_path):
    from pathlib import Path

    from qdc import atelier
    dossier = Path(__file__).parent / "donnees"
    noms = ["jeu_de_donnees_test_nettoyage_propre.csv", "jeu_de_donnees_test_nettoyage.xlsx"]
    tableaux = [atelier.charger((dossier / n).read_bytes(), n) for n in noms]
    communs = ingestion.identifiants_communs(tableaux)
    fichiers = []
    for n, df in zip(noms, tableaux):
        c = ingestion.deviner_correspondance(df)
        c.update({col: "id_client" for col in df.columns if col in communs})
        fichiers.append((n.split(".")[0], df, c))
    base = tmp_path / "q.duckdb"
    pipeline.executer_fichiers(fichiers, base)
    return base


def test_unicite_avant_traitement_compte_l_identifiant_commun(tmp_path):
    from qdc import diagnostic
    base = _base_frederic(tmp_path)
    d = diagnostic.diagnostiquer(pipeline.lire(base, "bronze_clients"), applicable=ingestion.applicable(base))
    assert dict(d.resume.select("dimension", "taux").iter_rows())["Unicité"] < 0.5  # 17 clients pour 37 fiches


def test_desaccords_entre_fichiers_et_date_future_signales(tmp_path):
    base = _base_frederic(tmp_path)
    gold = pipeline.lire(base, "gold_clients")
    pierre = gold.filter(pl.col("nom") == "Martin").row(0, named=True)
    assert any("Montant" in c and "1200" in c and "1012.2" in c for c in pierre["conflits"])
    assert not any("ID Client" in c for c in pierre["conflits"])
    silver = pipeline.lire(base, "silver_clients")
    assert silver.filter(pl.col("anomalies").list.contains("date de creation dans le futur")).height == 2


def test_fichiers_alignes_sur_la_fiche_unique(tmp_path):
    base = tmp_path / "q.duckdb"
    a = pl.DataFrame({"id": ["1"], "Prénom": ["Luccie"], "Nom": ["Pons"], "Tél": ["0612345678"]})
    b = pl.DataFrame({"id": ["1", "2"], "Prénom": ["Lucie", "Lucie"], "Nom": ["Pons", "Pons"],
                      "Tél": ["0612345678", "0612345678"]})
    c = pl.DataFrame({"id": ["1"], "Prénom": ["Lucie"], "Nom": ["Pons"], "Tél": ["0612345678"]})
    fichiers = [(n, df, ingestion.deviner_correspondance(df)) for n, df in (("A", a), ("B", b), ("C", c))]
    pipeline.executer_fichiers(fichiers, base)
    assert pipeline.fichier_corrige(base, "A")["Prénom"].item() == "Luccie"  # ses propres valeurs
    assert pipeline.fichier_corrige(base, "A", valeurs_referentiel=True)["Prénom"].item() == "Lucie"  # la fiche unique
