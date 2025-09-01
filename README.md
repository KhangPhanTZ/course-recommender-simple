# Coursera Content-Based Course Recommender

A GitHub-ready **content-based recommendation system** for Coursera-style datasets that only include **course metadata** (no user interactions). It provides:

- 🔎 TF-IDF & (optional) Sentence-BERT embeddings
- 🤝 Course-to-course similarity search
- 🧩 K-Means clustering + 2D visualization (UMAP)
- 🖥️ Streamlit app for demo (search by text or find similar courses)
- ⚙️ Config-driven column mapping (robust to unknown schema)

> Works out-of-the-box with `data/Coursera.csv`. If your column names differ, tweak `config/config.yaml` and rerun.

## Project Structure

```
.
├── app/
│   └── streamlit_app.py         # Streamlit UI
├── artifacts/                   # Saved models & vectors
├── config/
│   └── config.yaml              # Column names & settings
├── data/
│   └── Coursera.csv             # Your dataset (not committed by default)
├── notebooks/
│   └── EDA_guide.md             # Lightweight EDA guide
├── src/
│   ├── pipeline.py              # Orchestrates end-to-end build
│   ├── fe_text.py               # Text cleaning & feature building
│   ├── models/
│   │   ├── similarity.py        # Similarity search (TF-IDF or SBERT)
│   │   └── clustering.py        # KMeans + UMAP
│   └── utils/
│       ├── io.py                # IO helpers
│       └── config.py            # Config loader
├── tests/
│   └── smoke_test.py            # Quick pipeline sanity check
├── .gitignore
├── requirements.txt
└── README.md
```

## Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check/edit config**
   - Open `config/config.yaml`, confirm your column names (title, description, skills, etc.)

3. **Build artifacts (vectors, clusters)**
   ```bash
   python -m src.pipeline --mode build
   ```

4. **Run the app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## What it does

- Cleans & concatenates text fields → builds `corpus`
- Creates TF-IDF vectors (default) or Sentence-BERT embeddings (set `use_sbert: true`)
- Saves `artifacts/`:
  - `tfidf_vectorizer.pkl`, `X_tfidf.npz` (or `sbert_model_name.txt`, `X_sbert.npy`)
  - `kmeans.pkl`, `umap_embedding.npy`
  - `courses.parquet` (cleaned dataset + ids)
- Streamlit UI to:
  - Search by free text
  - Pick an existing course and get **Top-N similar** courses
  - Explore clusters + 2D map

## Notes

- If you don't have GPU, keep `use_sbert: false` for speed.
- Data is ignored by Git; upload your CSV or document how to obtain it.
- Replace Coursera.csv with your own and update config accordingly.

## Dataset 
- using dataset from kaggle : Multi-Platform Online Courses Dataset
- Dataset URL : https://www.kaggle.com/datasets/everydaycodings/multi-platform-online-courses-dataset
- Only coursera data in project operation
