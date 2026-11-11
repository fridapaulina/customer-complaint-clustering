import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

RESURSER = Path(__file__).parent / "resurser"

st.set_page_config(page_title="Ärendetyper i kundklagomål", layout="wide")


# ---------- Förbehandling (samma som i 01_eda.ipynb) ----------
def rensa_text(t):
    t = t.lower()
    t = re.sub(r"\bx{2,}\b", " ", t)       # maskering
    t = re.sub(r"\{\$[\d,.]+\}", " ", t)   # belopp
    t = re.sub(r"[^a-z\s]", " ", t)        # siffror och skiljetecken
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- Ladda modell och data ----------
@st.cache_resource
def ladda_modell():
    vec = joblib.load(RESURSER / "tfidf.joblib")
    lsa = joblib.load(RESURSER / "lsa.joblib")
    km = joblib.load(RESURSER / "kmeans14.joblib")
    klusternamn = joblib.load(RESURSER / "klusternamn.joblib")
    nyckelord = joblib.load(RESURSER / "nyckelord.joblib")
    return vec, lsa, km, klusternamn, nyckelord


@st.cache_data
def ladda_data():
    agg = pd.read_parquet(RESURSER / "aggregat.parquet")
    exempel = pd.read_parquet(RESURSER / "exempel.parquet")
    exempel_lsa = np.load(RESURSER / "exempel_lsa.npy")
    return agg, exempel, exempel_lsa


vec, lsa, km, klusternamn, nyckelord = ladda_modell()
agg, exempel, exempel_lsa = ladda_data()

per_typ = agg.groupby("arendetyp")[["antal", "fick_hjalp"]].sum()
per_typ["andel_hjalp"] = per_typ["fick_hjalp"] / per_typ["antal"]
snitt = agg["fick_hjalp"].sum() / agg["antal"].sum()

# ---------- Sidhuvud ----------
st.title("Ärendetyper i kundklagomål")
st.caption(
    "K-means-modell tränad på 101 373 klagomål till den amerikanska konsumentmyndigheten CFPB "
    "(nov 2022 – aug 2023). Modellen hittade 14 ärendetyper utan manuell märkning."
)

flik_sortera, flik_oversikt = st.tabs(["Sortera ett klagomål", "Översikt"])

# ---------- Flik 1: sortera ett klagomål ----------
with flik_sortera:
    if "text" not in st.session_state:
        st.session_state.text = ""

    if st.button("Hämta ett slumpmässigt exempel"):
        st.session_state.text = exempel["Consumer complaint narrative"].sample(1).iloc[0]

    st.text_area("Klistra in ett klagomål (på engelska)", key="text", height=200)

    if st.button("Sortera", type="primary"):
        ren = rensa_text(st.session_state.text)

        if len(ren.split()) < 10:
            st.warning("Skriv minst tio ord. Modellen tränades på klagomål med minst tio ord.")
        else:
            vektor = lsa.transform(vec.transform([ren]))
            avstand = km.transform(vektor)[0]          # avstånd till varje klustercentrum
            ordning = avstand.argsort()
            typ = klusternamn[ordning[0]]
            nast = klusternamn[ordning[1]]

            st.subheader(typ)
            st.write(f"Näst närmaste ärendetyp: *{nast}*")
            st.write("**Utmärkande ord för ärendetypen:** " + ", ".join(nyckelord[typ]))

            andel = per_typ.loc[typ, "andel_hjalp"]
            st.metric(
                "Andel kunder med denna ärendetyp som fick hjälp",
                f"{andel:.0%}",
                f"{(andel - snitt) * 100:+.1f} procentenheter jämfört med alla ärenden",
            )

            st.markdown("**Liknande klagomål**")
            likhet = exempel_lsa @ vektor[0]           # cosinuslikhet (vektorerna är normaliserade)
            kandidater = [i for i in likhet.argsort()[::-1] if likhet[i] < 0.999][:3]
            for i in kandidater:
                rad = exempel.iloc[i]
                hjalp = "fick hjälp" if rad["fick_hjalp"] == 1 else "fick inte hjälp"
                with st.expander(f"{rad['arendetyp']} · {rad['Product']} · {hjalp} · likhet {likhet[i]:.2f}"):
                    st.write(rad["Consumer complaint narrative"][:2000])

# ---------- Flik 2: översikt ----------
with flik_oversikt:
    c1, c2 = st.columns(2)
    produkter = c1.multiselect("Produkt", sorted(agg["Product"].unique()), placeholder="Alla produkter")
    foretag_lista = (
        ["Alla företag"]
        + sorted(f for f in agg["foretag"].unique() if f != "Övriga")
        + ["Övriga"]
    )
    foretag = c2.selectbox("Företag", foretag_lista)

    data = agg
    if produkter:
        data = data[data["Product"].isin(produkter)]
    if foretag != "Alla företag":
        data = data[data["foretag"] == foretag]

    if data.empty:
        st.info("Inga klagomål för det här urvalet.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Antal klagomål", f"{data['antal'].sum():,}".replace(",", " "))
        m2.metric("Andel som fick hjälp", f"{data['fick_hjalp'].sum() / data['antal'].sum():.1%}")

        typ_data = data.groupby("arendetyp")[["antal", "fick_hjalp"]].sum()
        typ_data["andel"] = typ_data["fick_hjalp"] / typ_data["antal"]

        g1, g2 = st.columns(2)

        fig, ax = plt.subplots(figsize=(6, 6))
        typ_data["antal"].sort_values().plot.barh(ax=ax, color="tab:blue")
        ax.set_title("Antal klagomål per ärendetyp")
        ax.set_ylabel("")
        plt.tight_layout()
        g1.pyplot(fig)

        visa = typ_data[typ_data["antal"] >= 30]
        fig, ax = plt.subplots(figsize=(6, 6))
        visa["andel"].sort_values().plot.barh(ax=ax, color="tab:green")
        ax.axvline(snitt, color="red", linestyle="--", label="Snitt, alla ärenden")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title("Andel som fick hjälp per ärendetyp")
        ax.set_ylabel("")
        ax.legend()
        plt.tight_layout()
        g2.pyplot(fig)
        g2.caption("Visar endast ärendetyper med minst 30 klagomål i urvalet.")

        st.markdown("**Utveckling över tid**")
        valda = st.multiselect(
            "Ärendetyper",
            sorted(typ_data.index),
            default=list(typ_data["antal"].nlargest(3).index),
        )
        if valda:
            tid = (
                data[data["arendetyp"].isin(valda)]
                .pivot_table(index="manad", columns="arendetyp", values="antal", aggfunc="sum")
                .fillna(0)
            )
            st.line_chart(tid)
            