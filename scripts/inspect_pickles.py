"""Inspect repository pickle artifacts (products.pkl, item_emb.pkl, etc.).

Run from PowerShell (no heredoc required):
    python scripts\inspect_pickles.py

Outputs a short report about which pickles load, the products DataFrame columns, sample rows,
and sample image values for detected image columns.
"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
files = ['products.pkl','item_emb.pkl','idx2item.pkl','item2idx.pkl','user_item_mat.pkl','user2idx.pkl']

print(f"Repository root: {ROOT}")
for fn in files:
    p = ROOT / fn
    if not p.exists():
        print(f"{fn}: MISSING")
        continue
    try:
        obj = joblib.load(p)
        t = type(obj)
        length = None
        try:
            length = getattr(obj, 'shape', None) or getattr(obj, '__len__', None)
        except Exception:
            length = None
        print(f"{fn}: OK (type={t}, len={length})")
    except Exception as e:
        print(f"{fn}: ERROR -> {e}")

# Inspect products DataFrame in more detail
p_products = ROOT / 'products.pkl'
if p_products.exists():
    try:
        df = joblib.load(p_products)
        if isinstance(df, pd.DataFrame):
            print('\nproducts.pkl: DataFrame detected')
            print('Columns:', list(df.columns))
            print('\nSample rows (first 5):')
            print(df.head(5).to_string())
            # detect likely image columns
            candidates = ['image_url','image','img','image_link','thumbnail','imageURL','product_image']
            found = [c for c in candidates if c in df.columns]
            if not found:
                # fuzzy detect
                found = [c for c in df.columns if 'image' in c.lower() or 'img' in c.lower()]
            print('\nDetected image columns (candidates):', found if found else 'None')
            if found:
                col = found[0]
                print(f'\nSample values from "{col}" column:')
                for i, v in enumerate(df[col].head(10).tolist(), start=1):
                    print(f" {i}. {repr(v)[:200]}")
        else:
            print('products.pkl loaded but is not a pandas DataFrame; type:', type(df))
    except Exception as e:
        print('Failed to load products.pkl ->', e)
else:
    print('products.pkl not found in repository root')
