# customer-complaint-clustering

## Installation

1. Klona repot:
   git clone https://github.com/fridapaulina/customer-complaint-clustering.git

2. Skapa och aktivera en virtuell miljö:
   python -m venv .venv
   source .venv/Scripts/activate    # Windows (Git Bash)
   source .venv/bin/activate        # Mac/Linux

3. Installera paket:
   pip install -r requirements.txt

## Data

Datan finns inte i repot på grund av storleken. Ladda ner
**CCDB Export November 2022 through August 2023** från
[CFPB:s Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
och lägg zip-filen (uppackas inte) i mappen `data/`.

Kör sedan notebooks i ordning, med början på `notebooks/01_eda.ipynb`.