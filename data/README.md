# Datasets

Drop real course catalogs here to build the full multi-platform catalog. The
build **auto-detects and merges** every recognized file — no config edits — and
falls back to the bundled synthetic sample (`examples/catalog/`) when this folder
is empty.

Detection is by column signature (not filename), but the loaders and `.gitignore`
are pre-wired for these three Kaggle datasets:

| Platform | Kaggle dataset | Expected file |
|----------|----------------|---------------|
| Coursera | [Coursera courses](https://www.kaggle.com/datasets/siddharthm1698/coursera-course-dataset) | `data/Coursera.csv` |
| Udemy    | [andrewmvd/udemy-courses](https://www.kaggle.com/datasets/andrewmvd/udemy-courses) | `data/udemy_courses.csv` |
| edX      | [imuhammad/edx-courses](https://www.kaggle.com/datasets/imuhammad/edx-courses) | `data/edx_courses.csv` |

## How to add them

This environment cannot reach Kaggle (the network policy blocks `kaggle.com`),
so download on your own machine and commit the files:

1. Download the CSVs from the links above (Kaggle account required).
2. Rename them to the filenames in the table (or keep any name — detection is by
   columns).
3. Put them in `data/` and commit. The listed filenames are un-ignored in
   `.gitignore` / `.dockerignore`, so they ship into the image.
4. Rebuild: `python scripts/build_demo_artifacts.py` (or rebuild the demo image).

To pull them *inside* a cloud session instead, allow `www.kaggle.com` in the
environment's network policy and set `KAGGLE_USERNAME` / `KAGGLE_KEY` as
environment secrets, then `pip install kaggle` and
`kaggle datasets download -d andrewmvd/udemy-courses`.

## Canonical schema

Every source is normalized onto:

```
id, title, provider, skills, category, level, rating, url, description, syllabus, source
```

Add support for a new platform by appending a `SourceSpec` in
[`src/data/sources.py`](../src/data/sources.py).

> Check each dataset's license before redistributing it in your repository/image.
