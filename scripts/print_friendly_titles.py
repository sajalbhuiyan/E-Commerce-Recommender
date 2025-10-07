import joblib
import pandas as pd
import re
from pathlib import Path

p = Path(__file__).parents[1] / 'products.pkl'
if not p.exists():
    print('products.pkl not found at', p)
    raise SystemExit(1)

df = joblib.load(p)

def friendly_title(row, pid=None):
    title = row.get('name') or row.get('title')
    if title and re.match(r"^\s*Product\s*\d+\s*$", str(title), re.IGNORECASE):
        brand = row.get('brand') or ''
        category = row.get('category') or ''
        price_val = None
        try:
            if row.get('price') is not None and not pd.isna(row.get('price')):
                price_val = float(row.get('price'))
        except Exception:
            price_val = None
        parts = []
        if brand and not pd.isna(brand):
            parts.append(str(brand))
        if category and not pd.isna(category):
            parts.append(str(category).title())
        if price_val is not None:
            parts.append(f"${price_val:.2f}")
        if parts:
            return ' • '.join(parts)
    return title or (f'Product {pid}' if pid is not None else 'Product')

for i, row in df.head(20).iterrows():
    pid = row.get('product_id')
    print(f"{i}. id={pid} name={row.get('name')!r} -> friendly={friendly_title(row,pid)!r} brand={row.get('brand')!r} price={row.get('price')}")
