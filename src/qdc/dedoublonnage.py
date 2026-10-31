"""Dedoublonnage (entity resolution) : quels enregistrements designent la meme personne ?

Le probleme : comparer chaque ligne a toutes les autres = 18 000 x 18 000 / 2 = 162 millions
de paires. Impossible a grande echelle. On procede donc en 4 temps :

1. BLOCAGE    : on ne compare que des lignes qui partagent au moins un indice
                (meme telephone, meme e-mail, debut du nom + code postal...).
                -> quelques dizaines de milliers de paires au lieu de 162 millions.
2. SCORE      : pour chaque paire, on additionne des points d'indices (meme date de naissance,
                noms tres proches...) et on en retire pour les contradictions (prenoms differents,
                civilites opposees...). Chaque point est EXPLIQUE.
3. DECISION   : score haut = doublon sur ; score bas = personnes differentes ;
                entre les deux = cas AMBIGU -> envoye a l'IA puis a l'humain (phase 4b).
4. GROUPES    : les doublons surs sont regroupes (A=B et B=C -> un seul client).

Et surtout : on MESURE la qualite du resultat contre la verite (precision / rappel).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from qdc import regles

SEUIL_DOUBLON = 7.0     # score >= 7 : meme personne, fusion automatique
SEUIL_DIFFERENT = 3.0   # score <  3 : personnes differentes
TAILLE_BLOC_MAX = 300   # un bloc trop gros (nom tres frequent) ne sert a rien et coute cher


# --- 1. Blocage ------------------------------------------------------------------
def preparer(silver: pl.DataFrame) -> pl.DataFrame:
    """Cles de comparaison : majuscules, sans accents (meme cle que le referentiel)."""
    k = regles.norm_ville
    return silver.select(
        "record_id", "source", "civilite", "email", "telephone", "code_postal", "date_naissance",
        k(pl.col("prenom")).alias("prenom"), k(pl.col("nom")).alias("nom"),
        k(pl.col("adresse")).alias("adresse"),
        pl.col("anomalies").list.contains("ordre prenom/nom incertain").alias("ordre_incertain"),
    )


BLOCS = {
    "telephone": pl.col("telephone"),
    "email": pl.col("email"),
    "nom+cp": pl.col("nom").str.slice(0, 4) + "|" + pl.col("code_postal"),
    "nom+initiale": pl.col("nom").str.slice(0, 4) + "|" + pl.col("prenom").str.slice(0, 1),
    "naissance+initiale": pl.col("date_naissance").cast(pl.String) + "|" + pl.col("prenom").str.slice(0, 1),
    # noms possiblement inverses (« Marie Timothee ») : on croise prenom et nom
    "croise+cp": pl.col("prenom").str.slice(0, 4) + "|" + pl.col("code_postal"),
}


def paires_candidates(p: pl.DataFrame) -> pl.DataFrame:
    morceaux = []
    for nom_bloc, cle in BLOCS.items():
        b = p.select("record_id", cle.alias("cle")).drop_nulls("cle")
        b = b.filter(pl.len().over("cle").is_between(2, TAILLE_BLOC_MAX))
        paires = b.join(b, on="cle", suffix="_b").filter(pl.col("record_id") < pl.col("record_id_b"))
        morceaux.append(paires.select(pl.col("record_id").alias("id_a"), pl.col("record_id_b").alias("id_b"),
                                      pl.lit(nom_bloc).alias("bloc")))
    toutes = pl.concat(morceaux)
    return toutes.group_by("id_a", "id_b").agg(pl.col("bloc").unique().sort().alias("blocs"))


# --- 2. Score explicable ------------------------------------------------------------
def _sim(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Similarite Jaro-Winkler (0 a 1), calculee en parallele ; 0 si une valeur manque."""
    s = cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(),
               scorer=JaroWinkler.normalized_similarity, workers=-1)
    return np.where(a.is_null().to_numpy() | b.is_null().to_numpy(), np.nan, s)


def scorer(paires: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    a = paires.join(p, left_on="id_a", right_on="record_id")
    ab = a.join(p, left_on="id_b", right_on="record_id", suffix="_b")

    direct = np.fmin(_sim(ab["nom"], ab["nom_b"]), 1)
    croise_nom = _sim(ab["nom"], ab["prenom_b"])
    croise_prenom = _sim(ab["prenom"], ab["nom_b"])
    sim_prenom_direct = _sim(ab["prenom"], ab["prenom_b"])
    # Si l'ordre prenom/nom est incertain d'un cote, on garde la meilleure lecture
    incertain = (ab["ordre_incertain"] | ab["ordre_incertain_b"]).to_numpy()
    croise_ok = incertain & (np.nan_to_num(croise_nom + croise_prenom) > np.nan_to_num(direct + sim_prenom_direct))
    sim_nom = np.where(croise_ok, croise_nom, direct)
    sim_prenom = np.where(croise_ok, croise_prenom, sim_prenom_direct)
    initiale = (ab["prenom"].str.contains(r"^[A-Z]\.$") | ab["prenom_b"].str.contains(r"^[A-Z]\.$")).fill_null(False)
    meme_initiale = (ab["prenom"].str.slice(0, 1) == ab["prenom_b"].str.slice(0, 1)).fill_null(False)

    # Attention : pour Polars, NaN est PLUS GRAND que tout nombre (NaN >= 0.97 est vrai !).
    # On convertit donc les NaN ("valeur absente") en null, qui rend toute comparaison fausse.
    serie = lambda x: pl.Series(x).fill_nan(None)  # noqa: E731
    ab = ab.with_columns(
        sim_nom=serie(sim_nom), sim_prenom=serie(sim_prenom),
        sim_adresse=serie(_sim(ab["adresse"], ab["adresse_b"])),
        prenom_initiale=(initiale & meme_initiale),
    )

    def regle(cond: pl.Expr, points: float, texte: str) -> pl.Expr:
        return pl.when(cond).then(pl.struct(points=pl.lit(points), raison=pl.lit(texte)))

    deux = lambda c: pl.col(c).is_not_null() & pl.col(f"{c}_b").is_not_null()  # noqa: E731
    egal = lambda c: deux(c) & (pl.col(c) == pl.col(f"{c}_b"))  # noqa: E731
    diff = lambda c: deux(c) & (pl.col(c) != pl.col(f"{c}_b"))  # noqa: E731

    indices = pl.concat_list([
        regle(egal("email"), 4, "même e-mail"),
        regle(egal("telephone"), 2.5, "même téléphone"),
        regle(egal("date_naissance"), 3, "même date de naissance"),
        regle(diff("date_naissance"), -4, "dates de naissance différentes"),
        regle(pl.col("sim_nom") >= 0.97, 2.5, "même nom"),
        regle(pl.col("sim_nom").is_between(0.88, 0.97, closed="left"), 1.5, "noms très proches (faute de frappe ?)"),
        regle(pl.col("sim_nom") < 0.75, -3, "noms différents"),
        regle(pl.col("sim_prenom") >= 0.97, 2, "même prénom"),
        regle(pl.col("sim_prenom").is_between(0.88, 0.97, closed="left"), 1, "prénoms très proches"),
        regle(pl.col("prenom_initiale"), 0.5, "initiale du prénom compatible"),
        regle((pl.col("sim_prenom") < 0.75) & ~pl.col("prenom_initiale"), -4, "prénoms différents"),
        regle(egal("code_postal"), 1, "même code postal"),
        regle(pl.col("sim_adresse") >= 0.9, 1.5, "même adresse"),
        regle(diff("civilite"), -3, "civilités opposées"),
    ]).list.drop_nulls()

    return ab.with_columns(indices=indices).with_columns(
        score=pl.col("indices").list.eval(pl.element().struct.field("points")).list.sum(),
    ).with_columns(
        decision=pl.when(pl.col("score") >= SEUIL_DOUBLON).then(pl.lit("doublon"))
        .when(pl.col("score") < SEUIL_DIFFERENT).then(pl.lit("different"))
        .otherwise(pl.lit("ambigu")),
    )


# --- 4. Groupes ------------------------------------------------------------------
def regrouper(ids: pl.Series, liens: pl.DataFrame) -> pl.DataFrame:
    """Composantes connexes : chaque enregistrement recoit l'identifiant de son client."""
    g = pl.DataFrame({"record_id": ids}).with_columns(client=pl.col("record_id"))
    for _ in range(50):
        prop = (liens.join(g, left_on="id_a", right_on="record_id")
                .join(g, left_on="id_b", right_on="record_id", suffix="_b")
                .select(pl.min_horizontal("client", "client_b").alias("m"), "id_a", "id_b"))
        mins = pl.concat([prop.select(pl.col("id_a").alias("record_id"), "m"),
                          prop.select(pl.col("id_b").alias("record_id"), "m")]).group_by("record_id").agg(pl.col("m").min())
        nouveau = g.join(mins, on="record_id", how="left").with_columns(
            client=pl.min_horizontal("client", "m")).drop("m")
        if nouveau.sort("record_id")["client"].equals(g.sort("record_id")["client"]):
            break
        g = nouveau
    return g


# --- Evaluation contre la verite ---------------------------------------------------------
@dataclass
class Evaluation:
    precision: float     # parmi les fusions faites, combien sont justes ?
    rappel: float        # parmi les vrais doublons, combien a-t-on trouves ?
    f1: float
    clients_estimes: int
    clients_reels: int
    fausses_fusions: int  # groupes qui melangent plusieurs vraies personnes


def evaluer(groupes: pl.DataFrame, verite: pl.DataFrame) -> Evaluation:
    g = groupes.join(verite.select("record_id", "person_id"), on="record_id")

    def nb_paires(col: str) -> int:  # nombre de paires d'enregistrements dans le meme groupe
        return g.group_by(col).len().select((pl.col("len") * (pl.col("len") - 1) / 2).sum()).item()

    predites, reelles = nb_paires("client"), nb_paires("person_id")
    justes = g.group_by("client", "person_id").len().select((pl.col("len") * (pl.col("len") - 1) / 2).sum()).item()
    precision = justes / predites if predites else 1.0
    rappel = justes / reelles if reelles else 1.0
    melanges = g.group_by("client").agg(pl.col("person_id").n_unique()).filter(pl.col("person_id") > 1).height
    return Evaluation(precision, rappel, 2 * precision * rappel / (precision + rappel),
                      g["client"].n_unique(), g["person_id"].n_unique(), melanges)


def executer(silver: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    p = preparer(silver)
    paires = scorer(paires_candidates(p), p)
    groupes = regrouper(p["record_id"], paires.filter(pl.col("decision") == "doublon").select("id_a", "id_b"))
    return paires, groupes


def afficher_paire(r: dict) -> str:
    raisons = ", ".join(f"{i['raison']} ({i['points']:+g})" for i in r["indices"])
    return (f"    {r['prenom']} {r['nom']} ({r['source']}) vs {r['prenom_b']} {r['nom_b']} ({r['source_b']})"
            f"  score {r['score']:+.1f}\n      -> {raisons}")


if __name__ == "__main__":
    import time

    base = Path("data/qdc.duckdb")
    with duckdb.connect(str(base)) as con:
        silver = con.sql("SELECT * FROM silver_clients").pl()

    t0 = time.perf_counter()
    paires, groupes = executer(silver)
    duree = time.perf_counter() - t0
    n = silver.height
    print(f"\n  BLOCAGE : {paires.height:,} paires comparees au lieu de {n * (n - 1) // 2:,}  ({duree:.1f} s)".replace(",", " "))
    print("\n  DECISIONS")
    for d, c in paires.group_by("decision").len().sort("decision").iter_rows():
        print(f"    {d:<10} {c:>7,}".replace(",", " "))

    ev = evaluer(groupes, pl.read_csv("data/verite/verite.csv"))
    print("\n  EVALUATION CONTRE LA VERITE (fusions automatiques seulement)")
    print(f"    precision {100 * ev.precision:5.1f} %   (parmi les fusions faites, combien sont justes)")
    print(f"    rappel    {100 * ev.rappel:5.1f} %   (parmi les vrais doublons, combien trouves)")
    print(f"    F1        {100 * ev.f1:5.1f} %")
    print(f"    clients estimes {ev.clients_estimes:,} / reels {ev.clients_reels:,}   groupes melanges : {ev.fausses_fusions}".replace(",", " "))

    print("\n  EXEMPLES DE CAS AMBIGUS (pour l'IA, phase 4b)")
    for r in paires.filter(pl.col("decision") == "ambigu").sample(4, seed=1).iter_rows(named=True):
        print(afficher_paire(r))

    with duckdb.connect(str(base)) as con:
        con.register("p", paires.select("id_a", "id_b", "blocs", "score", "decision", "indices").to_arrow())
        con.execute("CREATE OR REPLACE TABLE paires AS SELECT * FROM p")
        con.register("g", groupes.to_arrow())
        con.execute("CREATE OR REPLACE TABLE groupes_clients AS SELECT * FROM g")
