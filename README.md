# Automatisk upptäckt av ärendetyper i kundklagomål

Individuellt projekt i kursen Statistisk programmering och maskininlärning.

## Problem

Företag får tusentals kundklagomål i fritext. För att kunna bemanna rätt, utbilda personal och sortera ärenden behöver man veta vilka typer av ärenden som kommer in, men det finns sällan tid att läsa och märka upp texterna manuellt.

**Affärsfråga:** Kan vi med unsupervised learning hitta naturliga ärendetyper i kunders klagomålstexter, utan manuell märkning, och identifiera vilka ärendetyper som oftare leder till att kunden får hjälp?

## Resultat i korthet

- **K-means** delade in drygt 100 000 klagomål i **14 ärendetyper**, flera av dem på tvärs av myndighetens egna produktkategorier.
- Chansen att få hjälp skiljer sig tydligt mellan ärendetyperna: **störst** för obehöriga kortdebiteringar (30 %), **minst** för bolån (4 %).
- Klagomål som hänvisar till lagparagrafer, ofta i mallform, leder **sällan** till hjälp (11 %).
- En topp i januari 2023 sammanfaller med CFPB:s beslut mot Wells Fargo i december 2022.
- Tre metoder jämfördes: **K-means, LDA och HDBSCAN**. K-means valdes eftersom den bäst hittar nya ärendetyper, ger alla klagomål en ärendetyp och kan sortera in nya klagomål.

## Data

**Källa:** [CFPB Consumer Complaint Database – Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), den amerikanska konsumentskyddsmyndighetens öppna databas över klagomål på finansiella produkter.

Datan finns inte i repot på grund av storleken. Så här hämtar du den:

1. Ladda ner **CCDB Export November 2022 through August 2023** från länken ovan.
2. Lägg zip-filen i mappen `data/`. Den ska **inte** packas upp.

Filen heter `CCDB_Export_4_November_2022_through_August_2023.zip`. Datan hämtades 2026-11-02.

## Installation

Kräver Python 3.10 eller senare.

1. Klona repot:
   ```
   git clone https://github.com/fridapaulina/customer-complaint-clustering.git
   cd customer-complaint-clustering
   ```

2. Skapa och aktivera en virtuell miljö:
   ```
   python -m venv .venv
   source .venv/Scripts/activate    # Windows (Git Bash)
   .venv\Scripts\activate           # Windows (PowerShell/cmd)
   source .venv/bin/activate        # Mac/Linux
   ```

3. Installera paketen:
   ```
   pip install -r requirements.txt
   ```

`04_hdbscan.ipynb` laddar första gången ner embedding-modellen `all-MiniLM-L6-v2` (ca 90 MB) från Hugging Face.

## Notebooks

Kör notebooks i ordning. Varje notebook sparar sitt resultat i `data/` och nästa läser in det.

| Notebook | Innehåll | Ungefärlig körtid |
|---|---|---|
| `01_eda.ipynb` | Inläsning, rensning, EDA och statistiska tester | Några minuter |
| `02_kmeans.ipynb` | TF-IDF, LSA, val av antal kluster, K-means, namngivning, chi²-test | 5–10 min |
| `03_lda.ipynb` | Bag of Words och topic modelling med LDA, jämförelse med K-means | 5–10 min |
| `04_hdbscan.ipynb` | Embeddings, UMAP och HDBSCAN på 20 000 klagomål, jämförelse av alla metoder | 10–20 min första gången |
| `05_jamforelse.ipynb` | Val av metod, tolkning med beslutsträd, slutsatser | Några minuter |

## Metoder

- **Förbehandling:** textrensning (maskering, belopp, skiljetecken), egna stoppord (företagsnamn, allmänna klagomålsord)
- **Representation:** TF-IDF, LSA (TruncatedSVD), Bag of Words, sentence embeddings
- **Klustring:** K-means, LDA, UMAP + HDBSCAN
- **Utvärdering:** silhouette score, elbow-metoden, Adjusted Rand Index, manuell granskning
- **Statistik:** Mann-Whitney U, chi²-test med Cramér's V
- **Tolkning:** beslutsträd med feature importance

## Projektstruktur

```
customer-complaint-clustering/
├── data/            ← datafiler (ingår inte i repot)
├── notebooks/       ← 01–05, körs i ordning
├── app/             ← Streamlit-app
├── README.md
└── requirements.txt
```

## App

*Kommer.*

## Begränsningar

- Endast klagomål där konsumenten valt att publicera sin text ingår (39 %).
- Datan gäller amerikanska finansiella produkter, november 2022 – augusti 2023. Kreditupplysningsklagomål är borttagna.
- Utfallsvariabeln visar hur företaget avslutade ärendet, inte om kunden faktiskt hade rätt.

Se `05_jamforelse.ipynb` för fullständiga slutsatser och begränsningar.