import polars as pl
import pytest

from qdc import regles
from qdc.diagnostic import groupes_evidents


@pytest.mark.parametrize("brut", ["0612345678", "06 12 34 56 78", "06.12.34.56.78",
                                  "+33612345678", "+33 6 12 34 56 78", "0033612345678"])
def test_telephone_tous_formats(brut):
    out = pl.select(regles.norm_telephone(pl.lit(brut))).item()
    assert out == "+33612345678"


@pytest.mark.parametrize("brut", ["12345", "0612", "+44 20 7946 0958"])
def test_telephone_non_francais_rejete(brut):
    assert pl.select(regles.norm_telephone(pl.lit(brut))).item() is None


def test_statut_email():
    df = pl.DataFrame({"e": ["a.b@gmail.com", " A.B@Gmail.com ", "ab.gmail.com", None]})
    assert df.select(regles.statut_email("e")).to_series().to_list() == ["valide", "corrigeable", "invalide", "manquante"]


def test_cp_zero_initial_restaure():
    assert pl.select(regles.norm_cp(pl.lit("6000"))).item() == "06000"


def test_coherence_cp_ville_insensible_casse_et_accents():
    df = pl.DataFrame({"cp": ["25000", "25000", "99999"], "v": ["BESANCON", "Lyon", "Nulle-part"]})
    assert df.select(regles.cp_coherent("cp", "v")).to_series().to_list() == [True, False, None]


def test_groupes_transitifs():
    """A~B par e-mail, B~C par telephone => A, B, C dans le meme groupe ; D seul."""
    df = pl.DataFrame({
        "email": ["x@a.fr", " X@a.fr", "y@b.fr", "z@c.fr"],
        "telephone": ["0611111111", "06 22 22 22 22", "+33622222222", "0633333333"],
    })
    g = groupes_evidents(df).to_list()
    assert g[0] == g[1] == g[2] and g[3] != g[0]
