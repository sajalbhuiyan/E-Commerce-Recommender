"""Populate `products.pkl` with more suitable product names and exact image URLs.

This script replaces generic names like "Product 7" with names derived from
brand and category (e.g., "Brand 3 Wireless Headphones") and assigns a stable
internet image URL using loremflickr with a `lock` equal to the product numeric id
so the image remains consistent for each product.

Usage:
    python scripts/populate_names_and_images.py
"""
from pathlib import Path
import joblib
import pandas as pd
import re

ROOT = Path(__file__).parents[1]
P = ROOT / 'products.pkl'
if not P.exists():
    print('products.pkl not found at', P)
    raise SystemExit(1)

df = joblib.load(P)
if not isinstance(df, pd.DataFrame):
    print('products.pkl is not a pandas DataFrame')
    raise SystemExit(1)

df = df.copy()

# simple product type candidates by category
types_by_category = {
    'Electronics': ['Wireless Headphones','Bluetooth Speaker','Portable Charger','Smartwatch','Noise-Cancelling Headphones'],
    'Clothing': ['T-Shirt','Hoodie','Running Shoes','Jeans','Baseball Cap'],
    'Home': ['Ceramic Lamp','Throw Pillow','Kitchen Knife','Cutting Board','Wall Art'],
    'Books': ['Hardcover Novel','Pocket Guide','Cookbook','Children\'s Book','Art Book'],
    'Beauty': ['Moisturizing Cream','Fragrance','Lipstick','Facial Cleanser','Sunscreen'],
    'Sports': ['Yoga Mat','Fitness Tracker','Water Bottle','Tennis Racket','Running Shorts']
}

def numeric_id(pid):
    m = re.search(r"(\d+)", str(pid))
    return int(m.group(1)) if m else abs(hash(str(pid))) % 100000

def choose_type(cat, idx):
    if not cat or pd.isna(cat):
        cat = 'Electronics'
    cat = str(cat).title()
    pool = types_by_category.get(cat, types_by_category['Electronics'])
    return pool[idx % len(pool)]

for i, row in df.iterrows():
    pid = row.get('product_id')
    brand = row.get('brand') or ''
    category = row.get('category') if 'category' in df.columns else None
    if pd.isna(category):
        category = None

    # generate a nicer name when the existing name looks generic
    name = row.get('name')
    if not name or re.match(r"^\s*Product\s*\d+\s*$", str(name), re.IGNORECASE):
        ptype = choose_type(category, i)
        if brand and str(brand).strip():
            nice = f"{brand} {ptype}"
        else:
            nice = f"{ptype} {i}"
        df.at[i, 'name'] = nice

    # assign an exact/stable image using loremflickr with lock derived from pid
    num = numeric_id(pid)
    tag = (choose_type(category, i).split()[0]).lower() if category is not None else 'product'
    img_url = f"https://loremflickr.com/400/400/{tag}?lock={num}"
    df.at[i, 'image_url'] = img_url

joblib.dump(df, P)
print('Updated products.pkl with better names and exact image URLs (sample):')
print(df[['product_id','name','image_url']].head(12).to_string(index=False))
