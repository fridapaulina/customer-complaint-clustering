import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RESURSER = Path(__file__).parent / "resurser"
ACCENT = "#1F6F78"
GRA = "#B8C4CA"
ROD = "#C0392B"

st.set_page_config(page_title="Ärendetyper i kundklagomål", page_icon="📨", layout="wide")


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


def snygga_till(fig, hojd=480):
    fig.update_layout(
        height=hojd,
        margin=dict(l=0, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F2", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


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

    vanster, hoger = st.columns([3, 2], gap="large")

    with vanster:
        if st.button("Hämta ett slumpmässigt exempel"):
            st.session_state.text = exempel["Consumer complaint narrative"].sample(1).iloc[0]
        st.text_area("Klistra in ett klagomål (på engelska)", key="text", height=260)
        sortera = st.button("Sortera", type="primary", use_container_width=True)

    with hoger:
        if not sortera:
            st.info("Klistra in ett klagomål eller hämta ett exempel, och klicka på **Sortera**.")
        else:
            ren = rensa_text(st.session_state.text)
            if len(ren.split()) < 10:
                st.warning("Skriv minst tio ord. Modellen tränades på klagomål med minst tio ord.")
            else:
                vektor = lsa.transform(vec.transform([ren]))
                ordning = km.transform(vektor)[0].argsort()
                typ = klusternamn[ordning[0]]
                nast = klusternamn[ordning[1]]
                andel = per_typ.loc[typ, "andel_hjalp"]

                with st.container(border=True):
                    st.caption("ÄRENDETYP")
                    st.subheader(typ)
                    st.caption(f"Näst närmast: {nast}")
                    st.markdown(" ".join(f"`{o}`" for o in nyckelord[typ]))

                with st.container(border=True):
                    st.metric(
                        "Andel kunder med denna ärendetyp som fick hjälp",
                        f"{andel:.0%}",
                        f"{(andel - snitt) * 100:+.1f} procentenheter mot snittet ({snitt:.0%})",
                    )

    if sortera and len(rensa_text(st.session_state.text).split()) >= 10:
        st.markdown("#### Liknande klagomål")
        likhet = exempel_lsa @ vektor[0]           # cosinuslikhet (vektorerna är normaliserade)
        kandidater = [i for i in likhet.argsort()[::-1] if likhet[i] < 0.999][:3]
        for i in kandidater:
            rad = exempel.iloc[i]
            hjalp = "✅ fick hjälp" if rad["fick_hjalp"] == 1 else "❌ fick inte hjälp"
            with st.expander(f"{rad['arendetyp']}  ·  {rad['Product']}  ·  {hjalp}  ·  likhet {likhet[i]:.2f}"):
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
        m1, m2, m3 = st.columns(3)
        with m1.container(border=True):
            st.metric("Antal klagomål", f"{data['antal'].sum():,}".replace(",", " "))
        with m2.container(border=True):
            st.metric("Andel som fick hjälp", f"{data['fick_hjalp'].sum() / data['antal'].sum():.1%}")
        with m3.container(border=True):
            st.metric("Ärendetyper i urvalet", data["arendetyp"].nunique())

        typ_data = data.groupby("arendetyp")[["antal", "fick_hjalp"]].sum().reset_index()
        typ_data["andel"] = typ_data["fick_hjalp"] / typ_data["antal"]

        g1, g2 = st.columns(2)

        # Antal per ärendetyp
        antal_sorterad = typ_data.sort_values("antal")
        fig = px.bar(antal_sorterad, x="antal", y="arendetyp", orientation="h",
                     title="Antal klagomål per ärendetyp")
        fig.update_traces(marker_color=ACCENT,
                          hovertemplate="%{y}<br>%{x:,} klagomål<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        g1.plotly_chart(snygga_till(fig))

        # Andel som fick hjälp, färgad efter över/under snittet
        visa = typ_data[typ_data["antal"] >= 30].sort_values("andel").copy()
        visa["jamfort"] = np.where(visa["andel"] >= snitt, "Över snittet", "Under snittet")
        fig = px.bar(visa, x="andel", y="arendetyp", orientation="h", color="jamfort",
                     color_discrete_map={"Över snittet": ACCENT, "Under snittet": GRA},
                     custom_data=["antal"], title="Andel som fick hjälp per ärendetyp")
        fig.update_traces(hovertemplate="%{y}<br>%{x:.1%} fick hjälp<br>%{customdata[0]:,} klagomål<extra></extra>")
        fig.add_vline(x=snitt, line_dash="dash", line_color=ROD,
                      annotation_text=f"Snitt {snitt:.0%}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(categoryorder="array", categoryarray=list(visa["arendetyp"]))
        fig.update_layout(xaxis_title=None, yaxis_title=None,
                          legend=dict(orientation="h", y=-0.08))
        g2.plotly_chart(snygga_till(fig))
        g2.caption("Visar endast ärendetyper med minst 30 klagomål i urvalet.")

        # Utveckling över tid
        st.markdown("#### Utveckling över tid")
        valda = st.multiselect(
            "Välj ärendetyper",
            sorted(typ_data["arendetyp"]),
            default=list(typ_data.nlargest(3, "antal")["arendetyp"]),
        )
        if valda:
            tid = (
                data[data["arendetyp"].isin(valda)]
                .groupby(["manad", "arendetyp"])["antal"].sum()
                .reset_index()
            )
            fig = px.line(tid, x="manad", y="antal", color="arendetyp", markers=True,
                          color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,} klagomål<extra></extra>")
            fig.update_yaxes(rangemode="tozero")
            fig.update_layout(xaxis_title=None, yaxis_title="Antal klagomål",
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(snygga_till(fig, hojd=420))

            