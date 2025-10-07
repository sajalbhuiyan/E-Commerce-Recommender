"""Replace or populate the `image_url` column in products.pkl with stable
internet images using picsum.photos seeded by product_id. This helps the
Streamlit app display images without requiring local image files.

Usage:
    python scripts/update_images_to_picsum.py
"""
import joblib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]
P = ROOT / 'products.pkl'
if not P.exists():
    print('products.pkl not found at', P)
    raise SystemExit(1)

df = joblib.load(P)
if not isinstance(df, pd.DataFrame):
    print('products.pkl is not a pandas DataFrame')
    raise SystemExit(1)

def make_pic_url(pid, w=400, h=400):
    # picsum supports seeded images via the `seed` path
    # Using product_id ensures stable images per product
    return f'https://picsum.photos/seed/{pid}/{w}/{h}'

if 'product_id' not in df.columns:
    print('product_id column missing; cannot seed picsum by id')
    raise SystemExit(1)

df = df.copy()
df['image_url'] = df['product_id'].astype(str).apply(lambda pid: make_pic_url(pid))

joblib.dump(df, P)
print('Updated products.pkl with picsum image URLs (sample):')
print(df[['product_id','image_url']].head(10).to_string(index=False))
