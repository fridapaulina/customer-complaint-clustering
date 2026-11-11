import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RESOURCES = Path(__file__).parent / "resurser"
ACCENT = "#1F6F78"
GREY = "#B8C4CA"
RED = "#C0392B"

st.set_page_config(page_title="Customer Complaint Case Types", page_icon="📨", layout="wide")

# ---------- Translations (the model was trained with Swedish cluster names) ----------
NAMES = {
    "Bedrägeri och stängda bankkonton": "Fraud and closed bank accounts",
    "Spärrade eller försvunna pengar": "Frozen or missing funds",
    "Bristande ärendehantering": "Poor case handling",
    "Obehöriga kortdebiteringar": "Unauthorized card charges",
    "Juridiska hänvisningar och mallbrev": "Legal citations and template letters",
    "Ifrågasatta inkassokrav": "Disputed debt collection",
    "Sena betalningar och saldo": "Late payments and balances",
    "Räntor och lånevillkor": "Interest rates and loan terms",
    "Inkasso på kreditupplysningen": "Collections on credit reports",
    "Bolån: escrow och utmätning": "Mortgages: escrow and foreclosure",
    "Billån och återtagna bilar": "Auto loans and repossessions",
    "Felaktig kreditupplysning och identitetsstöld": "Credit report errors and identity theft",
    "Checkar och spärrade insättningar": "Checks and held deposits",
    "Avgifter och övertrassering": "Fees and overdrafts",
}

DESCRIPTIONS = {
    "Fraud and closed bank accounts": "Bank accounts closed by the bank or opened fraudulently, and fraud on checking or savings accounts.",
    "Frozen or missing funds": "The customer cannot access their own money: held funds, unexpected withdrawals or transfers that never arrived.",
    "Poor case handling": "Long back-and-forth with the company without a solution: unanswered calls, rejected disputes, promised documents that never arrive. The broadest type, mixing several issues.",
    "Unauthorized card charges": "Charges on credit or debit cards that the customer did not make, and disputes about card transactions.",
    "Legal citations and template letters": "Complaints citing laws such as 15 USC 1681 (FCRA), often written from templates and sometimes with pseudo-legal arguments.",
    "Disputed debt collection": "The customer disputes a debt collector's claim, asks for validation of the debt or says the debt is not theirs.",
    "Late payments and balances": "Disputes about late fees, how payments were posted and outstanding balances on cards and loans.",
    "Interest rates and loan terms": "Interest, loan modifications and how payments are applied, across student, mortgage, personal and auto loans.",
    "Collections on credit reports": "Collection accounts that appear on the customer's credit report.",
    "Mortgages: escrow and foreclosure": "Escrow accounts for taxes and insurance, loan modifications and foreclosure.",
    "Auto loans and repossessions": "Auto financing, contracts, vehicle titles and repossessed cars.",
    "Credit report errors and identity theft": "Incorrect information on credit reports and accounts opened through identity theft.",
    "Checks and held deposits": "Deposited checks that are held, returned or fraudulently cashed.",
    "Fees and overdrafts": "Overdraft fees and other bank fees the customer considers unfair.",
}

PRODUCT_SHORT = {
    "Checking or savings account": "Checking/savings",
    "Credit card or prepaid card": "Credit/prepaid card",
    "Debt collection": "Debt collection",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Mortgage": "Mortgage",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
}


# ---------- Preprocessing (same as in 01_eda.ipynb) ----------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"\bx{2,}\b", " ", t)       # masking (XXXX)
    t = re.sub(r"\{\$[\d,.]+\}", " ", t)   # amounts
    t = re.sub(r"[^a-z\s]", " ", t)        # digits and punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- Load model and data ----------
@st.cache_resource
def load_model():
    vec = joblib.load(RESOURCES / "tfidf.joblib")
    lsa = joblib.load(RESOURCES / "lsa.joblib")
    km = joblib.load(RESOURCES / "kmeans14.joblib")
    cluster_names = {k: NAMES[v] for k, v in joblib.load(RESOURCES / "klusternamn.joblib").items()}
    keywords = {NAMES[k]: v for k, v in joblib.load(RESOURCES / "nyckelord.joblib").items()}
    return vec, lsa, km, cluster_names, keywords


@st.cache_data
def load_data():
    agg = pd.read_parquet(RESOURCES / "aggregat.parquet")
    agg["case_type"] = agg["arendetyp"].map(NAMES)
    examples = pd.read_parquet(RESOURCES / "exempel.parquet")
    examples["case_type"] = examples["arendetyp"].map(NAMES)
    examples_lsa = np.load(RESOURCES / "exempel_lsa.npy")
    return agg, examples, examples_lsa


def style(fig, height=480):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F2", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


vec, lsa, km, cluster_names, keywords = load_model()
agg, examples, examples_lsa = load_data()

per_type = agg.groupby("case_type")[["antal", "fick_hjalp"]].sum()
per_type["help_rate"] = per_type["fick_hjalp"] / per_type["antal"]
average = agg["fick_hjalp"].sum() / agg["antal"].sum()

# ---------- Header ----------
st.title("Customer Complaint Case Types")
st.caption(
    "A K-means model trained on 101,373 complaints to the US Consumer Financial Protection Bureau "
    "(Nov 2022 – Aug 2023). The model found 14 case types without any manual labelling."
)

tab_classify, tab_overview, tab_about = st.tabs(["Classify a complaint", "Overview", "About the case types"])

# ---------- Tab 1: classify a complaint ----------
with tab_classify:
    if "text" not in st.session_state:
        st.session_state.text = ""

    left, right = st.columns([3, 2], gap="large")

    with left:
        if st.button("Load a random example"):
            st.session_state.text = examples["Consumer complaint narrative"].sample(1).iloc[0]
        st.text_area("Paste a complaint (in English)", key="text", height=260)
        classify = st.button("Classify", type="primary", use_container_width=True)

    cleaned = clean_text(st.session_state.text)
    valid = classify and len(cleaned.split()) >= 10

    with right:
        if not classify:
            st.info("Paste a complaint or load an example, then click **Classify**.")
        elif not valid:
            st.warning("Please write at least ten words. The model was trained on complaints with at least ten words.")
        else:
            vector = lsa.transform(vec.transform([cleaned]))
            order = km.transform(vector)[0].argsort()
            case_type = cluster_names[order[0]]
            runner_up = cluster_names[order[1]]
            rate = per_type.loc[case_type, "help_rate"]

            with st.container(border=True):
                st.caption("CASE TYPE")
                st.subheader(case_type)
                st.write(DESCRIPTIONS[case_type])
                st.caption(f"Second closest: {runner_up}")
                st.markdown(" ".join(f"`{w}`" for w in keywords[case_type]))

            with st.container(border=True):
                st.metric(
                    "Share of customers with this case type who received relief",
                    f"{rate:.0%}",
                    f"{(rate - average) * 100:+.1f} percentage points vs. average ({average:.0%})",
                )

    if valid:
        st.markdown("#### Similar complaints")
        similarity = examples_lsa @ vector[0]       # cosine similarity (vectors are normalized)
        candidates = [i for i in similarity.argsort()[::-1] if similarity[i] < 0.999][:3]
        for i in candidates:
            row = examples.iloc[i]
            helped = "✅ received relief" if row["fick_hjalp"] == 1 else "❌ no relief"
            with st.expander(f"{row['case_type']}  ·  {row['Product']}  ·  {helped}  ·  similarity {similarity[i]:.2f}"):
                st.write(row["Consumer complaint narrative"][:2000])

# ---------- Tab 2: overview ----------
with tab_overview:
    c1, c2 = st.columns(2)
    products = c1.multiselect("Product", sorted(agg["Product"].unique()), placeholder="All products")
    company_list = (
        ["All companies"]
        + sorted(f for f in agg["foretag"].unique() if f != "Övriga")
        + ["Other companies"]
    )
    company = c2.selectbox("Company", company_list)

    data = agg
    if products:
        data = data[data["Product"].isin(products)]
    if company == "Other companies":
        data = data[data["foretag"] == "Övriga"]
    elif company != "All companies":
        data = data[data["foretag"] == company]

    if data.empty:
        st.info("No complaints for this selection.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1.container(border=True):
            st.metric("Complaints", f"{data['antal'].sum():,}")
        with m2.container(border=True):
            st.metric("Share who received relief", f"{data['fick_hjalp'].sum() / data['antal'].sum():.1%}")
        with m3.container(border=True):
            st.metric("Case types in selection", data["case_type"].nunique())

        type_data = data.groupby("case_type")[["antal", "fick_hjalp"]].sum().reset_index()
        type_data["rate"] = type_data["fick_hjalp"] / type_data["antal"]

        g1, g2 = st.columns(2)

        # Complaints per case type
        fig = px.bar(type_data.sort_values("antal"), x="antal", y="case_type", orientation="h",
                     title="Complaints per case type")
        fig.update_traces(marker_color=ACCENT, hovertemplate="%{y}<br>%{x:,} complaints<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        g1.plotly_chart(style(fig))

        # Relief rate, coloured above/below average
        show = type_data[type_data["antal"] >= 30].sort_values("rate").copy()
        show["vs_average"] = np.where(show["rate"] >= average, "Above average", "Below average")
        fig = px.bar(show, x="rate", y="case_type", orientation="h", color="vs_average",
                     color_discrete_map={"Above average": ACCENT, "Below average": GREY},
                     custom_data=["antal"], title="Share who received relief, per case type")
        fig.update_traces(hovertemplate="%{y}<br>%{x:.1%} received relief<br>%{customdata[0]:,} complaints<extra></extra>")
        fig.add_vline(x=average, line_dash="dash", line_color=RED,
                      annotation_text=f"Average {average:.0%}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(categoryorder="array", categoryarray=list(show["case_type"]))
        fig.update_layout(xaxis_title=None, yaxis_title=None, legend=dict(orientation="h", y=-0.08))
        g2.plotly_chart(style(fig))
        g2.caption("Only case types with at least 30 complaints in the selection are shown.")

        # Heatmap: case type x product
        st.markdown("#### Relief rate by case type and product")
        st.caption(
            "The case type matters even within the same product. Cells with fewer than 50 complaints are left empty."
        )
        heat = data.groupby(["case_type", "Product"])[["antal", "fick_hjalp"]].sum().reset_index()
        heat = heat[heat["antal"] >= 50]
        if heat.empty:
            st.info("Too few complaints in this selection to show the heatmap.")
        else:
            heat["rate"] = heat["fick_hjalp"] / heat["antal"]
            heat["product"] = heat["Product"].map(PRODUCT_SHORT)
            pivot = heat.pivot(index="case_type", columns="product", values="rate")
            fig = px.imshow(pivot, text_auto=".0%", aspect="auto",
                            color_continuous_scale=["#F2F5F6", ACCENT], zmin=0, zmax=0.5)
            fig.update_traces(hovertemplate="%{y}<br>%{x}<br>%{z:.1%} received relief<extra></extra>")
            fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar=dict(title="", tickformat=".0%"))
            fig.update_xaxes(side="top", showgrid=False)
            st.plotly_chart(style(fig, height=560))

        # Trend over time
        st.markdown("#### Trend over time")
        selected = st.multiselect(
            "Select case types",
            sorted(type_data["case_type"]),
            default=list(type_data.nlargest(3, "antal")["case_type"]),
        )
        if selected:
            trend = (
                data[data["case_type"].isin(selected)]
                .groupby(["manad", "case_type"])["antal"].sum()
                .reset_index()
            )
            fig = px.line(trend, x="manad", y="antal", color="case_type", markers=True,
                          color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,} complaints<extra></extra>")
            fig.update_yaxes(rangemode="tozero")
            fig.update_layout(xaxis_title=None, yaxis_title="Complaints", legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(style(fig, height=420))

# ---------- Tab 3: about the case types ----------
with tab_about:
    st.markdown("#### The 14 case types")
    about = pd.DataFrame({
        "Case type": per_type.index,
        "Description": [DESCRIPTIONS[t] for t in per_type.index],
        "Keywords": [", ".join(keywords[t][:6]) for t in per_type.index],
        "Complaints": per_type["antal"].values,
        "Received relief": (per_type["help_rate"] * 100).values,
    }).sort_values("Complaints", ascending=False)

    st.dataframe(
        about,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Complaints": st.column_config.NumberColumn(format="%d"),
            "Received relief": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=50),
        },
    )

    with st.expander("How does the model work?", expanded=True):
        st.markdown(
            """
The model has read more than 100,000 complaints and grouped them into 14 case types by itself,
without anyone telling it which case types exist. It does this by looking at which words tend to
appear together: complaints that often mention *escrow*, *insurance* and *foreclosure* end up in one
group, complaints that mention *overdraft* and *fees* in another.

When you paste a new complaint, the model compares its words with each of the 14 groups and picks the
closest one. It also shows the second closest group, since many complaints are about more than one thing.

**Limitations**
- The model only knows complaints about US financial products from Nov 2022 – Aug 2023.
- *Poor case handling* is a broad group that collects complaints mainly describing a long process
  rather than a specific problem.
- "Received relief" shows how the company closed the case (monetary or other relief), not whether
  the customer was actually right.
"""
        )
        


            