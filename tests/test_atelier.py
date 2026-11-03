import io
from datetime import date

import polars as pl
import pytest

from qdc import atelier


def _ventes():
    return pl.DataFrame({
        "client": [" Dupont ", "Martin", "Martin", "Durand", "Petit", "Leroy", "Moreau", "Simon", "Laurent", "Roux"],
        "montant": ["1 234,56 €", "45,00", "45,00", "12,5", "60", "52,3", "48", "55,1", "47,9", "51"],
        "age": [34, 41, 41, None, 38, 212, 29, 45, 36, 40],
        "date": ["03/01/2025", "2025-01-05", "2025-01-05", "07/01/2025", "08.01.2025", "09/01/2025",
                 "10/01/2025", "11/01/2025", "12/01/2025", "13/01/2025"],
        "statut": ["payé", "N/A", "N/A", "payé", "impayé", "payé", "payé", "N/A", "payé", "payé"],
    })


def test_chargement_csv_francais_point_virgule_et_cp1252():
    contenu = "client;montant\nCélia;12,5\nNoé;3\n".encode("cp1252")
    df = atelier.charger(contenu, "export.csv")
    assert df.columns == ["client", "montant"] and df["client"].to_list() == ["Célia", "Noé"]


def test_format_refuse():
    with pytest.raises(ValueError, match="non pris en charge"):
        atelier.charger(b"x", "fichier.pdf")


def test_types_metier_detectes():
    df = _ventes()
    types = {c: atelier.type_colonne(df[c]) for c in df.columns}
    assert types == {"client": "texte", "montant": "nombre (texte)", "age": "nombre",
                     "date": "date (texte)", "statut": "texte"}


def test_nombres_francais_convertis():
    df = atelier.op_convertir_nombre(_ventes(), "montant").df
    assert df["montant"].to_list()[:4] == [1234.56, 45.0, 45.0, 12.5]


def test_dates_de_formats_varies():
    d = atelier.op_convertir_date(_ventes(), "date").df["date"].to_list()
    assert d[0] == date(2025, 1, 3) and d[1] == date(2025, 1, 5) and d[4] == date(2025, 1, 8)


def test_doublons_toutes_colonnes_ou_selection():
    df = _ventes()
    assert atelier.op_supprimer_doublons(df).n == 1
    assert atelier.op_supprimer_doublons(df, ["statut"]).n == 7  # il ne reste qu'une ligne par statut


def test_mediane_refusee_sur_du_texte():
    with pytest.raises(ValueError, match="numérique"):
        atelier.op_remplir_vides(_ventes(), "client", "mediane")


def test_remplir_vides_par_mediane_garde_le_type_entier():
    r = atelier.op_remplir_vides(_ventes(), "age", "mediane")
    assert r.n == 1 and r.df["age"].dtype == pl.Int64 and r.df["age"].null_count() == 0


def test_valeur_aberrante_detectee_puis_plafonnee():
    df = _ventes()
    signale = atelier.op_traiter_aberrantes(df, "age", "signaler")
    assert signale.n == 1 and signale.df.filter(pl.col("age_aberrante"))["age"].item() == 212
    plafonne = atelier.op_traiter_aberrantes(df, "age", "plafonner").df
    assert plafonne["age"].max() < 212 and plafonne.height == df.height


def test_remplacer_valeur_exacte_par_vide():
    r = atelier.op_remplacer(_ventes(), "statut", "N/A", "")
    assert r.n == 3 and r.df["statut"].null_count() == 3


def test_recette_rejouee_et_annulation():
    df = _ventes()
    recette = [{"op": "supprimer_doublons"},
               {"op": "nettoyer_texte", "colonnes": ["client"]},
               {"op": "convertir_nombre", "colonne": "montant"}]
    propre, journal = atelier.appliquer(df, recette)
    assert propre.height == 9 and propre["client"][0] == "Dupont" and propre["montant"].dtype == pl.Float64
    assert [e.n for e in journal][0] == 1
    # Annuler = rejouer sans la derniere etape, depuis l'original intact
    annule, _ = atelier.appliquer(df, recette[:-1])
    assert annule["montant"].dtype == pl.String and df["client"][0] == " Dupont "


def test_recette_reutilisable_le_mois_suivant():
    recette = [{"op": "convertir_nombre", "colonne": "montant"}]
    texte = atelier.recette_vers_json(recette)
    assert atelier.recette_depuis_json(texte) == recette
    mois_suivant = pl.DataFrame({"montant": ["10,5", "2 000"]})
    assert atelier.appliquer(mois_suivant, atelier.recette_depuis_json(texte))[0]["montant"].to_list() == [10.5, 2000.0]


def test_recette_sur_fichier_incompatible_message_clair():
    with pytest.raises(ValueError, match="Étape 1.*introuvable"):
        atelier.appliquer(pl.DataFrame({"x": [1]}), [{"op": "convertir_nombre", "colonne": "montant"}])


def test_propositions():
    titres = [p.titre for p in atelier.proposer(_ventes())]
    assert "Supprimer les doublons exacts" in titres
    assert any("montant" in t and "texte" in t for t in titres)
    assert any("vides" in t and "age" in t for t in titres)
    assert any("aberrantes" in t and "age" in t for t in titres)


def test_colonnes_compatibles_filtrees():
    df = atelier.op_convertir_nombre(_ventes(), "montant").df
    assert set(atelier.colonnes_compatibles(df, "traiter_aberrantes")) == {"montant", "age"}
