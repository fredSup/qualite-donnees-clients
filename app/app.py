"""Interface de l'outil : 6 ecrans, du fichier brut au referentiel client unique.

Lancer :  streamlit run app/app.py
L'interface ne contient AUCUNE regle metier : elle appelle le moteur (qdc.*), deja teste.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# En ligne (Streamlit Cloud), le moteur n'est pas installe comme bibliotheque : on indique ou il se trouve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import streamlit as st

from qdc import diagnostic, fusion, ia, pipeline, rapport

NOM_APP = "Unifia"
SOUS_TITRE = "Nettoyage et unification de données, assistés par l'IA"
DOSSIER = Path("data")
NOMS_SOURCES = {"crm": "CRM", "boutique": "Boutique en ligne", "support": "Service client"}
CHAMPS = {"civilite": "Civilité", "prenom": "Prénom", "nom": "Nom", "nom_complet": "Nom complet", "email": "E-mail",
          "telephone": "Téléphone", "adresse": "Adresse", "code_postal": "Code postal", "ville": "Ville",
          "date_naissance": "Date de naissance", "date_creation": "Date d'inscription",
          "date_dernier_contact": "Dernière activité", "id_client": "Identifiant client"}
OPERATIONS = {
    "telephone: format E.164": "Téléphone mis au format international",
    "nom: decoupe du nom complet": "Nom complet découpé en prénom + nom",
    "civilite: deduite du prenom": "Civilité déduite du prénom",
    "ville: deduite du code postal": "Ville déduite du code postal",
    "email: format": "E-mail remis en forme (espaces, majuscules)",
    "ordre prenom/nom incertain": "Ordre prénom / nom incertain",
    "email invalide (supprime)": "E-mail invalide (retiré)",
    "code postal incoherent avec la ville": "Code postal incohérent avec la ville",
    "date de naissance aberrante (supprimee)": "Date de naissance aberrante (retirée)",
    "date de creation dans le futur": "Date d'inscription dans le futur",
}
# Pour chaque anomalie : les colonnes a montrer pour la comprendre
COLONNES_ANOMALIE = {
    "ordre prenom/nom incertain": ["nom_complet", "prenom", "nom"],
    "email invalide (supprime)": ["email"],
    "code postal incoherent avec la ville": ["code_postal", "ville"],
    "date de naissance aberrante (supprimee)": ["date_naissance"],
    "date de creation dans le futur": ["date_creation"],
}

st.set_page_config(page_title=f"{NOM_APP} · qualité des données clients", page_icon="◆", layout="wide")


def espace_visiteur() -> Path:
    """Chaque visiteur a SON dossier : en ligne, un visiteur ne voit jamais les fichiers d'un autre.
    Les espaces de plus de 24 h sont effaces au passage (rien n'est conserve durablement)."""
    import shutil
    import time
    import uuid

    if "espace" not in st.session_state:
        racine = DOSSIER / "sessions"
        racine.mkdir(parents=True, exist_ok=True)
        for ancien in racine.iterdir():
            if ancien.is_dir() and time.time() - ancien.stat().st_mtime > 24 * 3600:
                shutil.rmtree(ancien, ignore_errors=True)
        st.session_state["espace"] = racine / uuid.uuid4().hex[:16]
    st.session_state["espace"].mkdir(parents=True, exist_ok=True)
    return st.session_state["espace"]


ESPACE = espace_visiteur()
RAW, BASE = ESPACE / "raw", ESPACE / "qdc.duckdb"

# Cle API : secrets Streamlit (en ligne) ou fichier .env (en local). Jamais dans le code.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:  # pas de fichier de secrets : normal en local
    pass

from ui import barre, carte, couleur, entete, ia_disponible, injecter_style, nb, pastille, pc  # noqa: E402

injecter_style()


def exiger_donnees() -> bool:
    if not pipeline.pret(BASE):
        st.info("Aucune donnée traitée pour l'instant. Commencez par l'écran **Source**.")
        return False
    return True


@st.cache_data(show_spinner=False)
def diagnostics(base: str, version_donnees: float):
    bronze = pipeline.lire(BASE, "bronze_clients")
    gold = pipeline.lire(BASE, "gold_clients")
    from qdc import ingestion
    avant = diagnostic.diagnostiquer(bronze, applicable=ingestion.applicable(BASE))
    df_g, applicable = fusion.pour_diagnostic(gold, ingestion.champs_collectes(BASE))
    apres = diagnostic.diagnostiquer(df_g, applicable=applicable)
    return avant, apres


def version() -> float:
    """Change a chaque ecriture de la base : invalide les caches."""
    return BASE.stat().st_mtime if BASE.exists() else 0.0


# --- 1. Source ----------------------------------------------------------------------
def ecran_accueil() -> None:
    st.markdown(f'<div class="u-hero"><h1><span class="marque">◆</span> {NOM_APP}</h1>'
                f'<p>{SOUS_TITRE}. Déposez un fichier : {NOM_APP} le diagnostique, vous propose des corrections '
                f'et explique chacune de ses décisions. Vous gardez la main, il fait le travail répétitif.</p></div>',
                unsafe_allow_html=True)
    st.markdown("### Que voulez-vous faire ?")
    g, d = st.columns(2)
    with g:
        with st.container(border=True):
            st.markdown("#### :material/cleaning_services: Nettoyer un fichier")
            st.markdown('<p class="u-note">Un tableau (ventes, clients, comptabilité, données Kaggle…) avec des '
                        'doublons, des cases vides, des valeurs aberrantes ou des formats mélangés.</p>',
                        unsafe_allow_html=True)
            st.markdown("- Diagnostic colonne par colonne\n- Corrections proposées, vous validez\n"
                        "- Médiane, moyenne, valeur choisie…\n- Fichier propre + recette réutilisable")
            st.page_link(PAGE_ATELIER, label="Ouvrir l'atelier", icon=":material/arrow_forward:")
    with d:
        with st.container(border=True):
            st.markdown("#### :material/join_inner: Unifier des fichiers clients")
            st.markdown('<p class="u-note">Les mêmes clients répartis dans plusieurs systèmes (CRM, boutique, '
                        'service client), écrits différemment, en double, parfois avec des fautes.</p>',
                        unsafe_allow_html=True)
            st.markdown("- Score de qualité sur 5 dimensions\n- Doublons retrouvés malgré les fautes\n"
                        "- L'IA et vous tranchez les cas douteux\n- Une fiche unique par client")
            st.page_link(PAGE_SOURCE, label="Unifier des fichiers", icon=":material/arrow_forward:")
    st.caption("Toutes les données de démonstration sont fictives. N'utilisez pas de données personnelles réelles "
               "sur la version en ligne.")


def ecran_source() -> None:
    entete("Unifier des fichiers clients", "Les mêmes clients dispersés dans plusieurs systèmes : l'outil les "
           "diagnostique, les nettoie et les réunit en une fiche unique par client.")

    c1, c2, c3 = st.columns(3)
    with c1:
        carte("1 · Mesurer", "Diagnostic", "5 dimensions de qualité, source par source")
    with c2:
        carte("2 · Corriger", "Nettoyage", "Ce qui est sûr est corrigé, le reste est signalé")
    with c3:
        carte("3 · Réunir", "Référentiel", "Doublons détectés, l'IA et vous tranchez les cas douteux")

    st.markdown("### Choisir les données")
    onglet_demo, onglet_fichiers = st.tabs(["Données de démonstration", "Vos fichiers"])
    with onglet_demo:
        st.markdown('<p class="u-note">Un jeu fictif et réaliste : 3 systèmes (CRM en CSV, boutique en Excel, '
                    'service client en Parquet) avec fautes de frappe, formats incohérents, doublons et pièges '
                    '(couples partageant un téléphone, e-mails pro…). Aucune donnée personnelle réelle.</p>',
                    unsafe_allow_html=True)
        n = st.select_slider("Nombre de personnes réelles", [1_000, 2_000, 5_000, 10_000, 20_000], value=10_000)
        if st.button("Générer et analyser", type="primary"):
            RAW.mkdir(parents=True, exist_ok=True)
            pipeline.generer_demo(RAW, n=n)
            lancer()
    with onglet_fichiers:
        ecran_fichiers()

    if pipeline.pret(BASE):
        st.success("Des données sont prêtes : parcourez les écrans dans le menu à gauche.")


def lancer(fichiers: list | None = None) -> None:
    with st.status("Traitement en cours…", expanded=True) as statut:
        try:
            ecrire = lambda m: statut.write(f"• {m}")  # noqa: E731
            r = (pipeline.executer_fichiers(fichiers, BASE, etape=ecrire) if fichiers
                 else pipeline.executer_tout(RAW, BASE, etape=ecrire))
        except ValueError as e:  # colonnes manquantes, limite depassee, format non pris en charge
            statut.update(label="Impossible de traiter ces fichiers", state="error")
            st.error(str(e))
            return
        statut.update(label=f"Terminé : {nb(r['fiches'])} fiches → {nb(r['clients'])} clients", state="complete")
    st.cache_data.clear()
    st.session_state.pop("file", None)


@st.cache_data(show_spinner=False, max_entries=20)
def lire_depot(contenu: bytes, nom: str) -> pl.DataFrame:
    from qdc import atelier
    return atelier.charger(contenu, nom)


def ecran_fichiers() -> None:
    """Depot de 1 a 10 fichiers quelconques, puis correspondance des colonnes devinee et corrigeable."""
    from qdc import ingestion
    st.markdown(f'<p class="u-note">Déposez de 1 à {ingestion.LIMITE_FICHIERS} exports (CSV, Excel ou Parquet, '
                f'jusqu\'à {nb(ingestion.LIMITE_LIGNES)} lignes au total). Pour chaque fichier, l\'outil devine ce '
                'que contient chaque colonne : vérifiez et corrigez si besoin. Les colonnes qu\'il ne reconnaît pas '
                '(montant, IBAN…) sont conservées et ajoutées à la fiche client.</p>', unsafe_allow_html=True)
    depots = st.file_uploader("Vos fichiers", type=["csv", "txt", "xlsx", "xls", "parquet"],
                              accept_multiple_files=True, label_visibility="collapsed")
    if not depots:
        return
    if len(depots) > ingestion.LIMITE_FICHIERS:
        st.error(f"{len(depots)} fichiers déposés : la limite est de {ingestion.LIMITE_FICHIERS}.")
        return
    options = list(ingestion.LIBELLES_CHAMPS.values()) + [ingestion.CONSERVER, ingestion.IGNORER]
    vers_champ = {v: k for k, v in ingestion.LIBELLES_CHAMPS.items()}
    tableaux = []
    for f in depots:
        try:
            tableaux.append(lire_depot(f.getvalue(), f.name))
        except Exception as e:
            st.error(f"{f.name} : impossible de lire ce fichier ({e}).")
            return
    communs = ingestion.identifiants_communs(tableaux)
    if communs:
        st.success(f"Identifiant commun détecté : « {', '.join(sorted(communs))} ». Deux fiches qui portent le même "
                   "identifiant seront réunies à coup sûr. Changez-le en « (conserver telle quelle) » si ce n'est pas "
                   "le même numéro client d'un fichier à l'autre.")
    prets = []
    for i, (f, df) in enumerate(zip(depots, tableaux)):
        with st.container(border=True):
            g, d = st.columns([1, 2.2])
            nom = g.text_input("Nom de la source", value=Path(f.name).stem[:30], key=f"src_nom_{i}",
                               help="Ex. CRM, Boutique, Compta : il apparaîtra dans les résultats.")
            g.caption(f"{f.name} · {nb(df.height)} lignes · {df.width} colonnes")
            devine = ingestion.deviner_correspondance(df)
            devine.update({c: "id_client" for c in df.columns if c in communs})
            tableau = pl.DataFrame({
                "Colonne du fichier": df.columns,
                "Exemple": [str(df[c].drop_nulls().head(1).to_list()[0]) if df[c].drop_nulls().len() else "—"
                            for c in df.columns],
                "Correspond à": [ingestion.LIBELLES_CHAMPS.get(devine[c], devine[c]) for c in df.columns],
            })
            edite = d.data_editor(tableau, hide_index=True, width="stretch", key=f"src_map_{i}",
                                  disabled=["Colonne du fichier", "Exemple"],
                                  column_config={"Correspond à": st.column_config.SelectboxColumn(
                                      "Correspond à", options=options, required=True)})
            corresp = {c: vers_champ.get(v, v) for c, v in zip(edite["Colonne du fichier"], edite["Correspond à"])}
            corresp = {c: v for c, v in corresp.items() if v != ingestion.IGNORER}
            prets.append((nom.strip() or f"Source {i + 1}", df, corresp))
    total = sum(df.height for _, df, _ in prets)
    st.caption(f"{len(prets)} fichier{'s' if len(prets) > 1 else ''} · {nb(total)} lignes au total")
    if st.button("Analyser mes fichiers", type="primary"):
        lancer(prets)


# --- 2. Diagnostic -------------------------------------------------------------------
def ecran_diagnostic() -> None:
    entete("Diagnostic qualité", "La qualité des données mesurée sur 5 dimensions, avant et après traitement. "
           "Un champ qu'une source ne collecte pas est « non applicable », pas « manquant ».")
    if not exiger_donnees():
        return
    avant, apres = diagnostics(str(BASE), version())
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown(f'<div class="u-carte"><div class="lib">Score de qualité</div><div class="u-score">'
                    f'<span class="val" style="color:#7A8C88">{str(avant.score_global).replace(".", ",")}</span><span class="u-fleche">→</span>'
                    f'<span class="val" style="color:{couleur(apres.score_global / 100)}">{str(apres.score_global).replace(".", ",")}</span>'
                    f'<span class="det">/ 100</span></div><div class="det">données brutes → référentiel</div></div>',
                    unsafe_allow_html=True)
    with c2:
        n_src = len(pipeline.sources(BASE)) or 3
        carte("Fiches analysées", nb(avant.n_lignes), f"dans {n_src} système{'s' if n_src > 1 else ''}")
    with c3:
        carte("Clients distincts", nb(apres.n_lignes), f"{100 * (1 - apres.n_lignes / avant.n_lignes):.0f} % de fiches en double")

    st.markdown("### Les 5 dimensions")
    a = dict(avant.resume.select("dimension", "taux").iter_rows())
    explications = {"Complétude": "la donnée est-elle renseignée ?", "Validité": "respecte-t-elle le format attendu ?",
                    "Cohérence": "code postal et ville concordent-ils ?", "Fraîcheur": "activité dans les 3 dernières années",
                    "Unicité": "chaque client n'apparaît qu'une fois"}
    g, d = st.columns(2)
    for i, (dim, t) in enumerate(apres.resume.select("dimension", "taux").iter_rows()):
        with (g if i % 2 == 0 else d):
            barre(f"<b>{dim}</b> · <span class='u-note'>{explications.get(dim, '')}</span>", t, a.get(dim))
    st.caption("L'unicité ne compte ici que les doublons évidents (même e-mail, même téléphone ou même identifiant client) : "
               "les cas douteux encore en attente n'y apparaissent pas.")

    st.markdown("### Par source, avant traitement")
    det = avant.details.filter(pl.col("source") != "toutes")
    pivot = (det.filter(pl.col("dimension").is_in(["Complétude", "Validité"]))
             .with_columns(pl.col("source").replace(NOMS_SOURCES),
                           (pl.col("dimension") + " · " + pl.col("champ").replace(CHAMPS)).alias("Indicateur"))
             .pivot(on="source", index="Indicateur", values="taux").sort("Indicateur")
             .with_columns(pl.exclude("Indicateur") * 100))
    st.dataframe(pivot, hide_index=True, width="stretch",
                 column_config={c: st.column_config.ProgressColumn(c, format="%.1f %%", min_value=0, max_value=100)
                                for c in pivot.columns if c != "Indicateur"})
    st.caption("Case vide : la source ne collecte pas ce champ (non applicable).")
    with st.expander("Exemples réels de problèmes détectés"):
        for k, v in avant.exemples.items():
            st.markdown(f"**{k}** : " + " · ".join(f"`{x}`" for x in v[:4]))


# --- 3. Nettoyage --------------------------------------------------------------------
def ecran_nettoyage() -> None:
    entete("Nettoyage", "Règle d'or : on corrige ce qui est sûr, on signale le reste, on ne devine jamais. "
           "Exemple : « 35000 REIMS » est signalé, car on ne sait pas qui a raison, du code postal ou de la ville.")
    if not exiger_donnees():
        return
    silver = pipeline.lire(BASE, "silver_clients")
    journal = (silver.select("source", pl.col("corrections").alias("op")).explode("op").drop_nulls()
               .with_columns(type=pl.lit("Corrigé")))
    anomalies = (silver.select("source", pl.col("anomalies").alias("op")).explode("op").drop_nulls()
                 .with_columns(type=pl.lit("Signalé")))
    tout = pl.concat([journal, anomalies]).with_columns(pl.col("source").replace(NOMS_SOURCES),
                                                       pl.col("op").replace(OPERATIONS))
    c1, c2, c3 = st.columns(3)
    with c1:
        carte("Corrections appliquées", nb(journal.height), "formats, découpage des noms, déductions sûres")
    with c2:
        carte("Problèmes signalés", nb(anomalies.height), "non corrigés automatiquement")
    with c3:
        carte("Fiches touchées", nb(silver.filter(pl.col("corrections").list.len() > 0).height),
              f"sur {nb(silver.height)}")

    st.markdown("### Détail par source")
    tableau = tout.group_by("type", "op", "source").len().pivot(on="source", index=["type", "op"], values="len") \
        .fill_null(0).sort(["type", "op"])
    st.dataframe(tableau.rename({"type": "Action", "op": "Opération"}), hide_index=True, width="stretch")

    st.markdown("### Avant / après, sur des fiches réelles")
    bronze = pipeline.lire(BASE, "bronze_clients")
    choix = st.segmented_control("Montrer", ["Téléphones", "E-mails", "Noms complets", "Anomalies"], default="Téléphones")
    filtres = {"Téléphones": pl.col("corrections").list.contains("telephone: format E.164"),
               "E-mails": pl.col("corrections").list.contains("email: format"),
               "Noms complets": pl.col("corrections").list.contains("nom: decoupe du nom complet"),
               "Anomalies": pl.col("anomalies").list.len() > 0}
    concernes = silver.filter(filtres[choix or "Téléphones"])
    if concernes.is_empty():
        st.info("Aucune fiche concernée par ce type de correction dans vos fichiers.")
        return
    ech = concernes.sample(min(8, concernes.height), seed=7, shuffle=True)
    if choix == "Anomalies":  # seulement les colonnes concernees par les anomalies presentes
        presentes = set(ech["anomalies"].explode().drop_nulls().to_list())
        cols = list(dict.fromkeys(c for a in presentes for c in COLONNES_ANOMALIE.get(a, [])))
    else:
        cols = {"Téléphones": ["telephone"], "E-mails": ["email"],
                "Noms complets": ["nom_complet", "prenom", "nom"]}[choix or "Téléphones"]
    av = bronze.select("record_id", *[pl.col(c).alias(f"{c} (avant)") for c in cols if c in bronze.columns])
    ap = ech.select("record_id", *[pl.col(c).cast(pl.String).alias(f"{c} (après)") for c in cols if c in ech.columns],
                    pl.col("anomalies").list.eval(pl.element().replace(OPERATIONS)).list.join(", ").alias("signalé"))
    vue = ap.join(av, on="record_id").select("record_id", *sorted(set(av.columns) - {"record_id"}),
                                             *[c for c in ap.columns if c != "record_id"])
    renommer = {"record_id": "Fiche"}
    for c in vue.columns:
        for k, v in CHAMPS.items():
            if c.startswith(k + " ("):
                renommer[c] = c.replace(k, v, 1)
    st.dataframe(vue.rename(renommer).rename({"signalé": "Signalé"}), hide_index=True, width="stretch")


# --- 4. Doublons et file de revue ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def file_de_revue(base: str, version_donnees: float) -> list[dict]:
    """Cas douteux, tries pour l'humain : d'abord ceux ou l'IA penche pour « meme personne »,
    puis ceux qu'elle n'a pas vus (par score), puis ceux qu'elle juge differents."""
    silver = pipeline.lire(BASE, "silver_clients")
    paires = pipeline.lire(BASE, "paires").filter(pl.col("decision") == "ambigu")
    deja = pipeline.decisions_humaines(BASE)
    paires = paires.join(deja.select("id_a", "id_b"), on=["id_a", "id_b"], how="anti")
    questions = ia.preparer_lot(paires.select("id_a", "id_b"), silver)
    cache = {r["cle"]: r for r in ia.lire_cache(BASE).iter_rows(named=True)}
    scores = {(r["id_a"], r["id_b"]): r for r in paires.iter_rows(named=True)}
    for q in questions:
        a, b = q["id"].split("|")
        q["score"], q["indices"] = scores[(a, b)]["score"], scores[(a, b)]["indices"]
        q["avis_ia"] = cache.get(q["_cle"])
        avis = q["avis_ia"]
        q["_rang"] = (0, -avis["confiance"]) if avis and avis["meme_personne"] else \
                     (1, -q["score"]) if not avis else (2, avis["confiance"])
    return sorted(questions, key=lambda q: q["_rang"])


def fiche_html(f: dict, comp: dict, titre: str) -> str:
    libelles = {"civilite": "Civilité", "prenom": "Prénom", "nom": "Nom", "email": "E-mail", "telephone": "Téléphone",
                "adresse": "Adresse", "code_postal": "Code postal", "ville": "Ville", "date_naissance": "Naissance"}
    genre = {"identique": "ok", "proche": "moy", "different": "ko", "inconnu": "neutre"}
    lignes = "".join(
        f'<tr><td class="c">{lib}</td><td>{f.get(c) or "<span class=u-note>—</span>"}</td>'
        f'<td style="text-align:right">{pastille(comp.get(c, ""), genre.get(comp.get(c, ""), "neutre"))}</td></tr>'
        for c, lib in libelles.items())
    return f'<div class="u-fiche"><b>{titre}</b><table>{lignes}</table></div>'


def ecran_doublons() -> None:
    entete("Doublons", "Les règles fusionnent seules les cas sûrs. Les cas douteux arrivent ici : "
           "l'IA donne son avis, vous décidez. Vos décisions priment sur tout.")
    if not exiger_donnees():
        return
    paires = pipeline.lire(BASE, "paires")
    decisions = dict(paires.group_by("decision").len().iter_rows())
    humaines = pipeline.decisions_humaines(BASE)
    file = file_de_revue(str(BASE), version())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carte("Paires comparées", nb(paires.height), "au lieu de toutes les combinaisons")
    with c2:
        carte("Fusions sûres", nb(decisions.get("doublon", 0)), "décidées par les règles")
    with c3:
        carte("Cas douteux", nb(decisions.get("ambigu", 0)), f"{nb(sum(1 for q in file if q['avis_ia']))} avec avis IA")
    with c4:
        carte("Vos décisions", nb(humaines.height), f"{nb(len(file))} cas restants")

    if not file:
        st.success("Plus aucun cas douteux en attente.")
        return
    i = min(st.session_state.get("file", 0), len(file) - 1)
    q = file[i]
    st.markdown(f"### Cas {i + 1} sur {nb(len(file))}")
    g, d = st.columns(2)
    sources = NOMS_SOURCES
    with g:
        st.markdown(fiche_html(q["fiche_1"], q["comparaison"], f"Fiche 1 · {sources.get(q['fiche_1']['source'], q['fiche_1']['source'])}"),
                    unsafe_allow_html=True)
    with d:
        st.markdown(fiche_html(q["fiche_2"], q["comparaison"], f"Fiche 2 · {sources.get(q['fiche_2']['source'], q['fiche_2']['source'])}"),
                    unsafe_allow_html=True)

    raisons = " · ".join(f"{x['raison']} ({x['points']:+g})" for x in q["indices"])
    st.markdown(f'<p class="u-note" style="margin-top:.8rem">Score des règles : <b>{q["score"]:+.1f}</b> '
                f'(doublon sûr à partir de 6) — {raisons}</p>', unsafe_allow_html=True)
    if q["comparaison"].get("prenom_et_nom_inverses") == "oui":
        st.markdown(pastille("prénom et nom inversés d'une fiche à l'autre", "moy"), unsafe_allow_html=True)

    avis = q["avis_ia"]
    if avis:
        verdict = "même personne" if avis["meme_personne"] else "personnes différentes"
        fiable = (avis["meme_personne"] and avis["confiance"] >= ia.CONFIANCE_FUSION) or \
                 (not avis["meme_personne"] and avis["confiance"] >= ia.CONFIANCE_SEPARATION)
        st.markdown(f'<div class="u-ia"><b>Avis de l\'IA</b> : {verdict} · confiance {avis["confiance"]:.2f} '
                    f'{pastille("appliqué automatiquement" if fiable and avis["meme_personne"] else "à confirmer", "ok" if fiable else "moy")}'
                    f'<br><span class="u-note">{avis["raison"]}</span></div>', unsafe_allow_html=True)
    elif ia_disponible():
        if st.button("Demander l'avis de l'IA"):
            with st.spinner("L'IA examine les deux fiches…"):
                d_ = ia.juger([q])
                ia.enregistrer(BASE, d_, {q["id"]: q["_cle"]})
            st.cache_data.clear()
            st.rerun()
    else:
        st.caption("Avis de l'IA indisponible (aucune clé API configurée).")

    st.write("")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    a, b = q["id"].split("|")
    if b1.button("Fusionner", type="primary", width="stretch"):
        pipeline.enregistrer_decision(BASE, a, b, "fusionner")
        st.cache_data.clear()
        st.rerun()
    if b2.button("Garder séparés", width="stretch"):
        pipeline.enregistrer_decision(BASE, a, b, "separer")
        st.cache_data.clear()
        st.rerun()
    if b3.button("Passer", width="stretch"):
        st.session_state["file"] = i + 1
        st.rerun()
    if b4.button(f"Mettre à jour le référentiel ({nb(humaines.height)} décisions)", width="stretch",
                 disabled=humaines.height == 0):
        with st.spinner("Reconstruction du référentiel…"):
            pipeline.construire_referentiel(BASE)
        st.cache_data.clear()
        st.toast("Référentiel mis à jour avec vos décisions.")


# --- 5. Referentiel -----------------------------------------------------------------------
COLONNES_GOLD = {"client_id", "id_client", "civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal", "ville",
                 "date_naissance", "date_premier_contact", "date_derniere_activite", "n_fiches", "sources",
                 "emails", "telephones", "variantes_nom", "conflits", "record_ids", "lignage"}


def colonnes_extra(gold: pl.DataFrame) -> list[str]:
    """Les colonnes conservees des fichiers d'origine (montant, IBAN, segment...)."""
    return [c for c in gold.columns if c not in COLONNES_GOLD]


def ecran_referentiel() -> None:
    entete("Référentiel client unique", "Une fiche par client, construite avec les meilleures valeurs de chaque "
           "système. Chaque champ garde la trace de la fiche d'où il vient.")
    if not exiger_donnees():
        return
    gold = pipeline.lire(BASE, "gold_clients")
    c1, c2, c3 = st.columns(3)
    with c1:
        carte("Clients", nb(gold.height), "dans le référentiel")
    with c2:
        carte("Réunis depuis plusieurs fiches", nb(gold.filter(pl.col("n_fiches") > 1).height), "")
    with c3:
        carte("Conflits à vérifier", nb(gold.filter(pl.col("conflits").list.len() > 0).height),
              "dates de naissance, désaccords entre fichiers…")

    # Le but de l'outil : repartir avec un fichier propre, a reimporter dans son CRM
    export = gold.select(
        "client_id", "civilite", "prenom", "nom", "email", "telephone", "adresse", "code_postal", "ville",
        "date_naissance", "date_premier_contact", "date_derniere_activite", "n_fiches",
        pl.col("sources").list.join(", "), pl.col("emails").list.join(", ").alias("tous_les_emails"),
        pl.col("telephones").list.join(", ").alias("tous_les_telephones"),
        pl.col("conflits").list.join(", "), pl.col("record_ids").list.join(", ").alias("fiches_d_origine"),
        *colonnes_extra(gold))
    st.write("")
    e1, e2, _ = st.columns([1, 1, 2])
    e1.download_button("Télécharger (CSV)", export.write_csv(separator=";"), "referentiel_clients.csv",
                       "text/csv", type="primary", width="stretch")
    tampon = io.BytesIO()
    export.write_excel(tampon)
    e2.download_button("Télécharger (Excel)", tampon.getvalue(), "referentiel_clients.xlsx", width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Télécharger chaque fichier d'origine, corrigé, avec son client_id"):
        st.caption("Chaque service récupère SON fichier, avec ses colonnes dans le même ordre, les valeurs "
                   "corrigées et, en première colonne, l'identifiant du client dans le référentiel. Deux fiches "
                   "qui désignent la même personne ont le même client_id, d'un fichier à l'autre.")
        aligner = st.toggle("Remplacer les valeurs par celles de la fiche unique", value=True,
                            help="Activé : chaque service reçoit exactement la même orthographe, le même e-mail "
                                 "principal, le même téléphone… Désactivé : chaque fichier garde ses propres "
                                 "valeurs (mises au bon format) et reçoit seulement le client_id.")
        cols = st.columns(min(len(pipeline.sources(BASE)), 4) or 1)
        for i, src in enumerate(pipeline.sources(BASE)):
            corrige = pipeline.fichier_corrige(BASE, src, valeurs_referentiel=aligner)
            cols[i % len(cols)].download_button(f"{NOMS_SOURCES.get(src, src)} ({nb(corrige.height)} lignes)",
                                                corrige.write_csv(separator=";"), f"{src}_corrige.csv",
                                                "text/csv", width="stretch", key=f"dl_{src}")

    g, d = st.columns([1.3, 1])
    with g:
        recherche = st.text_input("Rechercher un client", placeholder="Nom, prénom, e-mail ou ville")
        seulement = st.toggle("Seulement les clients avec un conflit")
        vue = gold
        if recherche:
            motif = "(?i)" + recherche.strip()
            vue = vue.filter(pl.any_horizontal(pl.col(c).str.contains(motif, literal=False).fill_null(False)
                                               for c in ("nom", "prenom", "email", "ville")))
        if seulement:
            vue = vue.filter(pl.col("conflits").list.len() > 0)
        vue = vue.sort("n_fiches", descending=True)
        sel = st.dataframe(vue.select(pl.col("client_id").alias("Client"), pl.col("prenom").alias("Prénom"),
                                      pl.col("nom").alias("Nom"), pl.col("email").alias("E-mail"),
                                      pl.col("ville").alias("Ville"), pl.col("n_fiches").alias("Fiches"),
                                      pl.col("sources").list.eval(pl.element().replace(NOMS_SOURCES))
                                      .list.join(", ").alias("Systèmes")).head(500),
                           hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=430)
        st.caption(f"{nb(vue.height)} clients · cliquez sur une ligne pour voir sa fiche")
    with d:
        lignes = sel.selection.rows if sel and sel.selection else []
        client = vue.row(lignes[0], named=True) if lignes else (vue.row(0, named=True) if vue.height else None)
        if client is None:
            st.info("Aucun client ne correspond.")
            return
        origine = client["lignage"] or {}
        champs = [("Identifiant client", "id_client", "origine_id_client"), ("Civilité", "civilite", "origine_civilite"), ("Prénom", "prenom", "origine_prenom"),
                  ("Nom", "nom", "origine_nom"), ("E-mail", "email", "origine_email"),
                  ("Téléphone", "telephone", "origine_telephone"), ("Adresse", "adresse", "origine_code_postal"),
                  ("Code postal", "code_postal", "origine_code_postal"), ("Ville", "ville", "origine_code_postal"),
                  ("Naissance", "date_naissance", "origine_date_naissance")]
        lignes_html = "".join(f'<tr><td class="c">{lib}</td><td>{client[c] if client[c] is not None else "—"}</td>'
                              f'<td class="o">{origine.get(o) or ""}</td></tr>'
                              for lib, c, o in champs)
        conflits = "".join(pastille(c, "ko") + " " for c in client["conflits"]) or pastille("aucun conflit", "ok")
        st.markdown(f'<div class="u-fiche"><b>{client["prenom"] or ""} {client["nom"] or ""}</b> '
                    f'<span class="u-note">· {client["client_id"]}</span><br>{conflits}'
                    f'<table style="margin-top:.5rem">{lignes_html}</table>'
                    f'<p class="u-note" style="margin:.6rem 0 0">Autres e-mails : {", ".join(client["emails"]) or "—"}<br>'
                    f'Fiches d\'origine : {", ".join(client["record_ids"])}</p>'
                    + (('<table style="margin-top:.5rem">' + "".join(
                        f'<tr><td class="c">{c}</td><td>{client[c] if client[c] is not None else "—"}</td></tr>'
                        for c in colonnes_extra(gold)) + "</table>") if colonnes_extra(gold) else "")
                    + '</div>', unsafe_allow_html=True)
        st.caption("À droite de chaque valeur : l'identifiant de la fiche d'origine (lignage).")


# --- 6. Rapport ---------------------------------------------------------------------------
def ecran_rapport() -> None:
    entete("Rapport d'audit", "Le code calcule les chiffres, l'IA rédige pour une lectrice non technique, "
           "puis le programme vérifie que chaque nombre cité existe bien dans les données.")
    if not exiger_donnees():
        return
    if not ia_disponible():
        st.caption("Sans clé API, le rapport est produit à partir d'un modèle de texte fixe.")
    if st.button("Générer le rapport", type="primary"):
        with st.spinner("Rédaction en cours…"):
            faits = rapport.collecter_faits(BASE)
            corps, auteur, suspects = rapport.rediger(faits)
            st.session_state["rapport"] = rapport.assembler(faits, corps, auteur, suspects)
    if "rapport" not in st.session_state:
        st.markdown('<div class="u-apercu"><b>Ce que contiendra le rapport</b><br>'
                    '1. Synthèse : le score de qualité avant → après<br>'
                    '2. État initial du fichier, source par source<br>'
                    '3. Ce qui a été corrigé<br>'
                    '4. Doublons : ce qui a été réuni<br>'
                    '5. Points de vigilance (cas douteux, clients inactifs, conflits)<br>'
                    '6. Recommandations concrètes, rattachées à chaque source</div>', unsafe_allow_html=True)
    if "rapport" in st.session_state:
        texte = st.session_state["rapport"]
        st.download_button("Télécharger (.md)", texte, file_name="rapport_audit.md", mime="text/markdown")
        with st.container(border=True):
            st.markdown(texte)


# --- Navigation ------------------------------------------------------------------------------
from page_atelier import page as page_atelier  # noqa: E402

PAGE_ACCUEIL = st.Page(ecran_accueil, title="Accueil", icon=":material/home:", url_path="accueil", default=True)
PAGE_ATELIER = st.Page(page_atelier, title="Atelier de nettoyage", icon=":material/cleaning_services:",
                       url_path="atelier")
PAGE_SOURCE = st.Page(ecran_source, title="Source", icon=":material/database:", url_path="source")
pages = {
    "": [PAGE_ACCUEIL],
    "Nettoyer un fichier": [PAGE_ATELIER],
    "Unifier des fichiers clients": [
        PAGE_SOURCE,
        st.Page(ecran_diagnostic, title="Diagnostic", icon=":material/monitoring:", url_path="diagnostic"),
        st.Page(ecran_nettoyage, title="Nettoyage", icon=":material/rule:", url_path="nettoyage"),
        st.Page(ecran_doublons, title="Doublons", icon=":material/join_inner:", url_path="doublons"),
        st.Page(ecran_referentiel, title="Référentiel", icon=":material/badge:", url_path="referentiel"),
        st.Page(ecran_rapport, title="Rapport", icon=":material/description:", url_path="rapport"),
    ],
}
st.navigation(pages).run()
with st.sidebar:
    st.caption("Projet de Frédéric Moukengue · Données 100 % fictives")
