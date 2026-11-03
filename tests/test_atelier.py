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
    assert any("inhabituelles" in t and "age" in t for t in titres)


def test_colonnes_compatibles_filtrees():
    df = atelier.op_convertir_nombre(_ventes(), "montant").df
    assert set(atelier.colonnes_compatibles(df, "traiter_aberrantes")) == {"montant", "age"}


def test_csv_avec_dates_de_formats_melanges_se_charge():
    """Bug trouve en preparant les tests de Frederic : le chargement plantait."""
    contenu = "ref;date\nA;03/02/2025\nB;2025-02-04\nC;05.02.2025\n".encode("utf-8")
    df = atelier.charger(contenu, "commandes.csv")
    assert df.height == 3 and atelier.type_colonne(df["date"]) == "date (texte)"


def _commandes():
    from pathlib import Path
    f = Path(__file__).parent / "donnees" / "test_mode1_commandes.csv"
    return atelier.charger(f.read_bytes(), f.name)


def test_regles_connues_reconnues_au_nom_de_la_colonne():
    df = pl.DataFrame({"Age client": [30, 40], "Panier": [10.0, 20.0], "Ville": ["a", "b"]})
    r = atelier.regles_par_defaut(df)
    assert (r["Age client"]["min"], r["Age client"]["max"]) == (0, 120)
    assert r["Panier"]["min"] == 0 and "Ville" not in r


def test_fichier_de_test_mode1_impossible_et_inhabituel_distingues():
    """Le corrige : ages 245, -4, 999 IMPOSSIBLES ; le panier de 120 EUR n'est PAS une anomalie."""
    df, _ = atelier.appliquer(_commandes(), [{"op": "supprimer_doublons"},
                                             {"op": "convertir_nombre", "colonne": "Panier"}])
    props = atelier.proposer(df, atelier.regles_par_defaut(df))
    impossibles = next(p for p in props if p.titre == "Valeurs impossibles dans « Age »")
    assert impossibles.n == 3 and list(impossibles.choix)[0] == "Vider"
    inhab_panier = [p for p in props if p.titre == "Valeurs inhabituelles dans « Panier »"]
    assert all("120" not in p.explication for p in inhab_panier)   # plus de faux positif sur 120 EUR
    vide, _ = atelier.appliquer(df, [impossibles.choix["Vider"]])
    assert vide["Age"].null_count() == 2 + 3 and vide["Age"].max() <= 120   # rien d'invente


def test_une_colonne_traitee_n_est_plus_reproposee():
    df = pl.DataFrame({"age": [30, 31, 32, 33, 34, 35, 36, 37, 38, 250]})
    recette = [{"op": "traiter_aberrantes", "colonne": "age", "action": "plafonner"}]
    propre, _ = atelier.appliquer(df, recette)
    deja = {(o["op"], o.get("colonne")) for o in recette}
    assert not [p for p in atelier.proposer(propre, deja=deja) if "age" in p.titre and "inhabituelles" in p.titre]


def test_dates_dans_le_futur():
    df = pl.DataFrame({"inscription": [date(2024, 1, 1), date(2099, 1, 1)]})
    assert any("futur" in p.titre for p in atelier.proposer(df))
    assert atelier.op_dates_futures(df, "inscription", "vider").df["inscription"].null_count() == 1


def test_regles_proposees_par_l_ia_sont_controlees():
    import json
    df = pl.DataFrame({"Age": [30, 40], "Note": [3, 4]})
    rep = json.dumps({"regles": [{"colonne": "Age", "min": 0, "max": 120, "raison": "âge humain"},
                                 {"colonne": "Inventee", "min": 0, "max": 1},          # colonne inexistante
                                 {"colonne": "Note", "min": 10, "max": 1}]})            # min > max
    r = atelier.regles_ia(df, lambda m: rep)
    assert list(r) == ["Age"] and r["Age"]["source"] == "IA"


def test_valeur_de_remplacement_a_la_precision_de_la_colonne():
    """Bug trouve par Frederic : un panier rempli par la moyenne valait 69.83500000000001."""
    df = pl.DataFrame({"panier": [45.9, 120.0, 38.5, 64.2, 81.35, None]})
    r = atelier.op_remplir_vides(df, "panier", "moyenne")
    assert r.df["panier"][-1] == round((45.9 + 120.0 + 38.5 + 64.2 + 81.35) / 5, 2)
