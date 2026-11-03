"""Mode 1 : l'atelier de nettoyage d'un fichier (4 etapes : Charger, Diagnostic, Atelier, Exporter).

Etat conserve dans la session :
- at_origine : le fichier d'origine (jamais modifie)
- at_recette : la liste des operations appliquees
- at_ignorees : les propositions que l'utilisateur a ecartees
Le fichier courant n'est jamais stocke : il est toujours recalcule = appliquer(origine, recette).
"""

from __future__ import annotations

import io

import polars as pl
import streamlit as st

from qdc import atelier
from ui import carte, entete, nb, pastille

TYPES = {"nombre": "Nombre", "nombre (texte)": "Nombre écrit en texte", "date": "Date",
         "date (texte)": "Date écrite en texte", "email": "E-mail", "telephone": "Téléphone",
         "texte": "Texte", "booleen": "Oui / non", "autre": "Autre"}
METHODES = {"mediane": "Médiane", "moyenne": "Moyenne", "plus_frequente": "Valeur la plus fréquente",
            "valeur": "Une valeur que je choisis"}
ACTIONS = {"signaler": "Signaler (ajoute une colonne d'alerte)", "plafonner": "Plafonner (ramener aux bornes)",
           "supprimer": "Supprimer les lignes"}
CASSES = {"aucune": "Ne pas changer", "majuscules": "MAJUSCULES", "minuscules": "minuscules", "titre": "Majuscule En Tête"}


def _etat() -> dict:
    s = st.session_state
    s.setdefault("at_recette", [])
    s.setdefault("at_ignorees", set())
    return s


@st.cache_data(show_spinner=False, max_entries=4)
def _courant(_origine: pl.DataFrame, cle: str, recette_json: str):
    """Fichier courant = origine + recette. Mis en cache : rejouer 10 operations sur 100 000 lignes
    prend une fraction de seconde, mais inutile de le refaire a chaque clic."""
    return atelier.appliquer(_origine, atelier.recette_depuis_json(recette_json))


def courant():
    s = _etat()
    return _courant(s["at_origine"], s["at_cle"], atelier.recette_vers_json(s["at_recette"]))


def ajouter(operation: dict) -> None:
    s = _etat()
    essai = s["at_recette"] + [operation]
    try:
        atelier.appliquer(s["at_origine"], essai)
    except ValueError as e:
        st.session_state["at_erreur"] = str(e)
        return
    s["at_recette"] = essai
    st.session_state.pop("at_erreur", None)


def nouveau_fichier(df: pl.DataFrame, nom: str) -> None:
    s = _etat()
    s["at_origine"], s["at_nom"], s["at_cle"] = df, nom, f"{nom}-{df.height}-{df.width}-{hash(tuple(df.columns))}"
    s["at_recette"], s["at_ignorees"] = [], set()


# --- 1. Charger -------------------------------------------------------------------------
def etape_charger() -> None:
    s = _etat()
    g, d = st.columns([1.4, 1])
    with g:
        fichier = st.file_uploader("Déposez votre fichier", type=["csv", "txt", "xlsx", "xls", "parquet"],
                                   help="CSV (séparateur détecté automatiquement), Excel ou Parquet.")
        if fichier is not None and s.get("at_nom") != fichier.name:
            try:
                nouveau_fichier(atelier.charger(fichier.getvalue(), fichier.name), fichier.name)
            except Exception as e:  # fichier illisible : message clair plutot qu'une trace technique
                st.error(f"Impossible de lire ce fichier : {e}")
    with d:
        st.markdown('<p class="u-note">Pas de fichier sous la main ? Essayez avec un export de ventes fictif '
                    '(5 000 lignes) : doublons, montants « 1 234,56 € », dates en plusieurs formats, '
                    'âges aberrants, « N/A »…</p>', unsafe_allow_html=True)
        if st.button("Utiliser le fichier d'exemple", width="stretch"):
            nouveau_fichier(atelier.exemple_ventes(), "exemple_ventes.csv")
        with st.expander("Réutiliser une recette"):
            st.caption("Une recette enregistrée le mois dernier ? Chargez-la pour refaire tout le nettoyage d'un coup.")
            rec = st.file_uploader("Recette (.json)", type=["json"], key="at_recette_fichier",
                                   label_visibility="collapsed")
            if rec is not None and "at_origine" in s and st.button("Appliquer la recette"):
                try:
                    operations = atelier.recette_depuis_json(rec.getvalue().decode("utf-8"))
                    atelier.appliquer(s["at_origine"], operations)
                    s["at_recette"] = operations
                    st.success(f"{len(operations)} opérations appliquées.")
                except (ValueError, UnicodeDecodeError) as e:
                    st.error(str(e))

    if "at_origine" in s:
        df = s["at_origine"]
        st.markdown(f"**{s['at_nom']}** · {nb(df.height)} lignes · {df.width} colonnes")
        st.dataframe(df.head(50), hide_index=True, width="stretch", height=280)
        if st.button("Étape suivante : le diagnostic →", type="primary"):
            st.session_state["at_suivante"] = "2 · Diagnostic"
            st.rerun()


# --- 2. Diagnostic --------------------------------------------------------------------------
def etape_diagnostic() -> None:
    df, _ = courant()
    profils, n_doublons = atelier.profiler(df)
    total_vides = sum(p.n_vides for p in profils)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carte("Lignes", nb(df.height), f"{df.width} colonnes")
    with c2:
        carte("Doublons exacts", nb(n_doublons), "lignes identiques")
    with c3:
        carte("Cellules vides", nb(total_vides), f"{100 * total_vides / max(df.height * df.width, 1):.1f} % du tableau".replace(".", ","))
    with c4:
        carte("Valeurs aberrantes", nb(sum(p.n_aberrantes for p in profils)), "colonnes numériques")

    st.markdown("### Colonne par colonne")
    tableau = pl.DataFrame([{
        "Colonne": p.nom, "Contenu détecté": TYPES.get(p.type, p.type), "Vides (%)": round(100 * p.pct_vides, 1),
        "Valeurs distinctes": p.n_distincts, "Aberrantes": p.n_aberrantes or None,
        "Espaces en trop": p.n_espaces or None, "Exemples": " · ".join(p.exemples),
    } for p in profils])
    st.dataframe(tableau, hide_index=True, width="stretch",
                 column_config={"Vides (%)": st.column_config.ProgressColumn("Vides", format="%.1f %%",
                                                                              min_value=0, max_value=100)})
    st.caption("« Nombre écrit en texte » : des montants comme « 1 234,56 € », illisibles pour un calcul "
               "tant qu'ils ne sont pas convertis. Les valeurs aberrantes sont repérées par la méthode de "
               "l'écart interquartile.")


# --- 3. Atelier ---------------------------------------------------------------------------
def formulaire_operation(df: pl.DataFrame) -> None:
    libelles = {k: v[1] for k, v in atelier.OPERATIONS.items()}
    op = st.selectbox("Opération", list(libelles), format_func=libelles.get, key="at_op")
    compat = atelier.colonnes_compatibles(df, op)
    if op != "supprimer_doublons" and not compat:
        st.caption("Aucune colonne compatible avec cette opération.")
        return
    params: dict = {}
    if op == "supprimer_doublons":
        params["colonnes"] = st.multiselect("Comparer seulement ces colonnes (vide = toutes)", list(df.columns)) or None
    elif op in ("supprimer_lignes_vides", "supprimer_colonnes", "nettoyer_texte"):
        params["colonnes"] = st.multiselect("Colonnes", compat)
        if op == "nettoyer_texte":
            params["casse"] = st.radio("Majuscules", list(CASSES), format_func=CASSES.get, horizontal=True)
    else:
        params["colonne"] = st.selectbox("Colonne (seules les compatibles sont proposées)", compat)
        if op == "remplir_vides":
            numerique = atelier.type_colonne(df[params["colonne"]]) == "nombre"
            choix = list(METHODES) if numerique else ["plus_frequente", "valeur"]
            params["methode"] = st.radio("Remplacer par", choix, format_func=METHODES.get, horizontal=True)
            if params["methode"] == "valeur":
                params["valeur"] = st.text_input("Valeur")
            if not numerique:
                st.caption("Médiane et moyenne : seulement pour les colonnes numériques.")
        elif op == "traiter_aberrantes":
            params["action"] = st.radio("Action", list(ACTIONS), format_func=ACTIONS.get)
        elif op == "remplacer":
            params["rechercher"] = st.text_input("Valeur exacte à remplacer", placeholder="N/A")
            params["remplacer"] = st.text_input("Remplacer par (vide = laisser vide)")
        elif op == "renommer":
            params["nouveau"] = st.text_input("Nouveau nom")
    pret = all(v not in (None, [], "") for k, v in params.items() if k not in ("colonnes", "remplacer", "valeur")) \
        and (op != "supprimer_colonnes" or params.get("colonnes"))
    if op in ("supprimer_lignes_vides", "nettoyer_texte"):
        pret = pret and bool(params.get("colonnes"))
    if st.button("Ajouter l'opération", type="primary", disabled=not pret, width="stretch"):
        ajouter({"op": op, **{k: v for k, v in params.items() if v is not None}})
        st.rerun()


def etape_atelier() -> None:
    s = _etat()
    df, journal = courant()
    if s.get("at_erreur"):
        st.error(s["at_erreur"])
    g, d = st.columns([1.55, 1])
    with g:
        props = [p for p in atelier.proposer(df) if p.titre not in s["at_ignorees"]]
        st.markdown(f"#### Propositions de l'outil · {len(props)}")
        if not props:
            st.success("Plus rien à proposer : le fichier semble propre. Vous pouvez encore ajouter vos propres "
                       "opérations à droite, ou passer à l'export.")
        for i, p in enumerate(props[:8]):
            with st.container(border=True):
                st.markdown(f"**{p.titre}** &nbsp; {pastille(nb(p.n), 'moy')}", unsafe_allow_html=True)
                st.markdown(f'<p class="u-note">{p.explication}</p>', unsafe_allow_html=True)
                boutons = st.columns(len(p.choix) + 1)
                for j, (libelle, operation) in enumerate(p.choix.items()):
                    if boutons[j].button(libelle, key=f"at_p{i}_{j}", type="primary" if j == 0 else "secondary"):
                        ajouter(operation)
                        st.rerun()
                if boutons[-1].button("Ignorer", key=f"at_p{i}_ign"):
                    s["at_ignorees"].add(p.titre)
                    st.rerun()
        if len(props) > 8:
            st.caption(f"… et {len(props) - 8} autres propositions après celles-ci.")
    with d:
        with st.container(border=True):
            st.markdown("#### Ajouter une opération")
            formulaire_operation(df)
        with st.container(border=True):
            st.markdown(f"#### Journal · {len(journal)} opération{'s' if len(journal) > 1 else ''}")
            if not journal:
                st.caption("Aucune opération pour l'instant.")
            for k, e in enumerate(journal, start=1):
                st.markdown(f"{k}. {e.message} {pastille(nb(e.n), 'ok')}", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("Annuler la dernière", disabled=not journal, width="stretch"):
                s["at_recette"] = s["at_recette"][:-1]
                st.rerun()
            if c2.button("Tout recommencer", disabled=not journal, width="stretch"):
                s["at_recette"], s["at_ignorees"] = [], set()
                st.rerun()
    st.markdown(f"#### Aperçu du fichier · {nb(df.height)} lignes")
    st.dataframe(df.head(30), hide_index=True, width="stretch", height=260)


# --- 4. Exporter ---------------------------------------------------------------------------
def etape_exporter() -> None:
    s = _etat()
    df, journal = courant()
    origine = s["at_origine"]
    c1, c2, c3 = st.columns(3)
    with c1:
        carte("Lignes", nb(df.height), f"au lieu de {nb(origine.height)}")
    with c2:
        avant = sum(p.n_vides for p in atelier.profiler(origine)[0])
        apres = sum(p.n_vides for p in atelier.profiler(df)[0])
        carte("Cellules vides", nb(apres), f"au lieu de {nb(avant)}")
    with c3:
        carte("Opérations", str(len(journal)), "rejouables avec la recette")

    base = s["at_nom"].rsplit(".", 1)[0]
    st.markdown("### Télécharger")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Fichier propre (CSV)", df.write_csv(separator=";"), f"{base}_propre.csv",
                       "text/csv", type="primary", width="stretch")
    tampon = io.BytesIO()
    df.write_excel(tampon)
    d2.download_button("Fichier propre (Excel)", tampon.getvalue(), f"{base}_propre.xlsx", width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    rapport = "\n".join([f"# Journal de nettoyage · {s['at_nom']}", "",
                         f"{nb(origine.height)} lignes au départ, {nb(df.height)} à l'arrivée.", ""]
                        + [f"{k}. {e.message} ({nb(e.n)})" for k, e in enumerate(journal, start=1)])
    d3.download_button("Journal (.md)", rapport, f"{base}_journal.md", "text/markdown", width="stretch")
    d4.download_button("Recette (.json)", atelier.recette_vers_json(s["at_recette"]), f"{base}_recette.json",
                       "application/json", width="stretch", disabled=not journal)
    st.caption("La recette contient toutes vos opérations : le mois prochain, chargez le nouvel export puis "
               "la recette (étape Charger) pour tout refaire en un clic.")
    with st.expander("Voir le journal"):
        st.markdown(rapport)


ETAPES = {"1 · Charger": etape_charger, "2 · Diagnostic": etape_diagnostic,
          "3 · Atelier": etape_atelier, "4 · Exporter": etape_exporter}


def page() -> None:
    entete("Nettoyer un fichier", "Déposez un fichier, l'outil le diagnostique et vous propose des corrections. "
           "Vous décidez, il applique. Le fichier d'origine n'est jamais modifié.")
    # Un selecteur (et non des onglets) : il garde l'etape choisie quand la page se recharge
    # apres un clic, au lieu de renvoyer l'utilisateur sur la premiere etape.
    st.session_state.setdefault("at_etape", "1 · Charger")
    if "at_suivante" in st.session_state:  # changer de widget par code : AVANT sa creation
        st.session_state["at_etape"] = st.session_state.pop("at_suivante")
    choix = st.segmented_control("Étape", list(ETAPES), key="at_etape", label_visibility="collapsed")
    choix = choix or "1 · Charger"
    st.write("")
    if choix != "1 · Charger" and "at_origine" not in st.session_state:
        st.info("Chargez d'abord un fichier à l'étape **1 · Charger**.")
        return
    ETAPES[choix]()
