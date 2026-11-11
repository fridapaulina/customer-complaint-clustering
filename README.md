# Customer Complaint Case Types

Individual project in the course *Statistical Programming and Machine Learning*.

**Live app:** [https://customer-complaint-case-types.streamlit.app/](https://customer-complaint-case-types.streamlit.app/)

## Problem

Companies receive thousands of customer complaints as free text. To staff correctly, train employees and route cases, they need to know what types of cases are coming in, but there is rarely time to read and label the texts manually.

**Business question:** Can unsupervised learning find natural case types in customer complaints, without manual labelling, and identify which case types more often lead to the customer receiving relief?

## Key findings

- **K-means** grouped 101,373 complaints into **14 case types**, several of which cut across the regulator's own product categories.
- The chance of receiving relief differs clearly between case types: **highest** for unauthorized card charges (30 %), **lowest** for mortgages (4 %).
- Complaints citing laws, often written from templates, **rarely** lead to relief (11 %).
- A spike in January 2023 coincides with the CFPB's order against Wells Fargo in December 2022.
- Three methods were compared: **K-means, LDA and HDBSCAN**. K-means was chosen because it best finds new case types, assigns every complaint a case type and can classify new complaints.

## The app

The app has three tabs:
- **Classify a complaint:** paste a complaint and get its case type, keywords, relief rate and similar complaints.
- **Overview:** complaints and relief rates per case type, a heatmap of case type by product, and trends over time, filterable by product and company.
- **About the case types:** a description of all 14 case types and a non-technical explanation of the model.

To run the app locally (no data download needed, the model and summary data are included in `app/resurser/`):
```
streamlit run app/app.py
```

## Data

**Source:** [CFPB Consumer Complaint Database – Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), the US Consumer Financial Protection Bureau's open database of complaints about financial products.

The data is not included in the repository due to its size. To run the notebooks:

1. Download **CCDB Export November 2022 through August 2023** from the link above.
2. Place the zip file in the `data/` folder. Do **not** unzip it.

The file is named `CCDB_Export_4_November_2022_through_August_2023.zip`. Data retrieved 2026-11-02.

## Installation

Requires Python 3.10 or later.

1. Clone the repository:
   ```
   git clone https://github.com/fridapaulina/customer-complaint-clustering.git
   cd customer-complaint-clustering
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/Scripts/activate    # Windows (Git Bash)
   .venv\Scripts\activate           # Windows (PowerShell/cmd)
   source .venv/bin/activate        # Mac/Linux
   ```

3. Install the packages:
   ```
   pip install -r requirements.txt
   ```

`04_hdbscan.ipynb` downloads the embedding model `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face the first time it runs.

## Notebooks

The notebooks are written in Swedish. Run them in order: each notebook saves its results to `data/` and the next one reads them.

| Notebook | Content | Approx. run time |
|---|---|---|
| `01_eda.ipynb` | Loading, cleaning, EDA and statistical tests | A few minutes |
| `02_kmeans.ipynb` | TF-IDF, LSA, choosing the number of clusters, K-means, naming, chi² test, saving the model for the app | 5–10 min |
| `03_lda.ipynb` | Bag of Words and topic modelling with LDA, comparison with K-means | 5–10 min |
| `04_hdbscan.ipynb` | Embeddings, UMAP and HDBSCAN on 20,000 complaints, comparison of all methods | 10–20 min first time |
| `05_jamforelse.ipynb` | Choice of method, interpretation with a decision tree, conclusions | A few minutes |

## Methods

- **Preprocessing:** text cleaning (masking, amounts, punctuation), custom stop words (company names, generic complaint words)
- **Representation:** TF-IDF, LSA (TruncatedSVD), Bag of Words, sentence embeddings
- **Clustering:** K-means, LDA, UMAP + HDBSCAN
- **Evaluation:** silhouette score, elbow method, Adjusted Rand Index, manual review
- **Statistics:** Mann-Whitney U, chi² test with Cramér's V
- **Interpretation:** decision tree with feature importance

## Project structure

```
customer-complaint-clustering/
├── .streamlit/config.toml   ← app theme
├── app/
│   ├── app.py               ← Streamlit app
│   ├── requirements.txt     ← packages for the app only (used by Streamlit Cloud)
│   └── resurser/            ← trained model and summary data
├── data/                    ← data files (not in the repository)
├── notebooks/               ← 01–05, run in order
├── README.md
└── requirements.txt         ← packages for the whole project
```

## Limitations

- Only complaints where the consumer chose to publish their text are included (39 %).
- The data covers US financial products, November 2022 – August 2023. Credit reporting complaints were excluded.
- "Received relief" shows how the company closed the case, not whether the customer was actually right.

See `05_jamforelse.ipynb` for full conclusions and limitations.