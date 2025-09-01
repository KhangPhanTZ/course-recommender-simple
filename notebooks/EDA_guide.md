# EDA Guide (Lightweight)

Suggested quick checks:
- Shape, columns, missing rates
- Value counts: category, level
- Text length distribution for title/description
- Rating distribution (if exists)
- Skills parsing: how many skills per course? Top-20 skills

Example (Notebook/Python):
```python
import pandas as pd
df = pd.read_csv('data/Coursera.csv')
print(df.shape, df.columns.tolist())
df['title_len'] = df['title'].astype(str).str.len()
df['desc_len']  = df['description'].astype(str).str.len()
print(df[['title_len','desc_len']].describe())
```
