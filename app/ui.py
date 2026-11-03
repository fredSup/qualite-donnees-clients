"""Composants graphiques communs aux deux modes : un seul design pour tout l'outil."""

from __future__ import annotations

import os
import textwrap

import streamlit as st


def injecter_style() -> None:
    # dedent : en Markdown, une ligne indentee de 4 espaces deviendrait un bloc de code
    st.markdown(textwrap.dedent("""
    <style>
    :root { --vert:#0F6E56; --sombre:#12201F; --teinte:#E7F0EE; --ambre:#B26A00; --rouge:#B42318; }
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    .u-hero { background: var(--sombre); color: #fff; border-radius: 14px; padding: 1.6rem 1.8rem; margin-bottom: 1.4rem; }
    .u-hero h1 { color:#fff; font-size: 1.9rem; margin: 0 0 .3rem 0; }
    .u-hero p { color: #C9DAD5; margin: 0; font-size: 1.02rem; }
    .u-hero .marque { color: #7FD1B9; }
    .u-carte { background: #fff; border: 1px solid #DCE7E3; border-radius: 12px; padding: 1rem 1.15rem; height: 100%; }
    .u-carte .lib { color: #4A5E5A; font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }
    .u-carte .val { font-size: 1.9rem; font-weight: 650; color: var(--sombre); line-height: 1.2; }
    .u-carte .det { color: #4A5E5A; font-size: .88rem; }
    .u-score { display:flex; align-items:baseline; gap:.6rem; }
    .u-fleche { color:#7A8C88; font-size: 1.4rem; }
    .u-barre { background: var(--teinte); border-radius: 99px; height: 10px; overflow: hidden; margin: .35rem 0 .8rem; }
    .u-barre > div { height: 100%; border-radius: 99px; }
    .u-dim { display:flex; justify-content: space-between; font-size: .93rem; }
    .u-pastille { display:inline-block; padding: .12rem .55rem; border-radius: 99px; font-size: .78rem; font-weight: 600; }
    .p-ok { background:#E3F4EC; color:#0B5C46; } .p-moy { background:#FFF1D6; color:#7A4A00; } .p-ko { background:#FDE7E5; color:#8F1D14; }
    .p-neutre { background:#EEF2F1; color:#3D4F4B; }
    .u-fiche { border: 1px solid #DCE7E3; border-radius: 12px; padding: .9rem 1.1rem; background:#fff; }
    .u-fiche table { width:100%; border-collapse: collapse; font-size: .92rem; }
    .u-fiche td { padding: .28rem .2rem; border-bottom: 1px solid #F0F4F3; vertical-align: top; }
    .u-fiche td.c { color:#5B6E6A; width: 34%; }
    .u-ia { border-left: 4px solid var(--vert); background: var(--teinte); padding: .8rem 1rem; border-radius: 0 10px 10px 0; }
    .u-note { color:#5B6E6A; font-size: .88rem; }
    .u-fiche td.o { text-align:right; color:#7A8C88; font-size:.76rem; white-space:nowrap; font-family: ui-monospace, monospace; }
    .u-apercu { border: 1px dashed #B9CEC8; border-radius: 12px; padding: 1rem 1.2rem; color:#4A5E5A; background:#FAFCFB; }
    /* Boutons d'acces aux modes (page d'accueil) */
    [data-testid="stPageLink-NavLink"] { background: var(--vert); border-radius: 8px; padding: .5rem 1.1rem;
        width: fit-content; transition: background .15s, transform .15s, box-shadow .15s; }
    [data-testid="stPageLink-NavLink"] * { color: #fff !important; font-weight: 600; }
    [data-testid="stPageLink-NavLink"]:hover { background: #0B5443; transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15,110,86,.28); }
    /* Menu de gauche : survol et page active un peu plus marques */
    [data-testid="stSidebarNavLink"] { transition: background .15s; }
    [data-testid="stSidebarNavLink"]:hover { background: #D3E7E1 !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] { background: #C3DED6 !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] * { color: var(--vert) !important; font-weight: 600; }
    </style>
    """), unsafe_allow_html=True)


# --- Petits composants ------------------------------------------------------------------
def couleur(t: float) -> str:
    return "#0F6E56" if t >= 0.95 else ("#C98A12" if t >= 0.8 else "#B42318")


def carte(libelle: str, valeur: str, detail: str = "") -> None:
    st.markdown(f'<div class="u-carte"><div class="lib">{libelle}</div><div class="val">{valeur}</div>'
                f'<div class="det">{detail}</div></div>', unsafe_allow_html=True)


def pc(t: float) -> str:
    return f"{100 * t:.1f} %".replace(".", ",")


def barre(libelle: str, t: float, avant: float | None = None) -> None:
    droite = pc(t) if avant is None else f"{pc(avant)} → <b>{pc(t)}</b>"
    st.markdown(f'<div class="u-dim"><span>{libelle}</span><span>{droite}</span></div>'
                f'<div class="u-barre"><div style="width:{100 * t:.1f}%;background:{couleur(t)}"></div></div>',
                unsafe_allow_html=True)


def pastille(texte: str, genre: str = "neutre") -> str:
    return f'<span class="u-pastille p-{genre}">{texte}</span>'


def nb(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def entete(titre: str, texte: str) -> None:
    st.markdown(f"## {titre}")
    st.markdown(f'<p class="u-note">{texte}</p>', unsafe_allow_html=True)


def ia_disponible() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


