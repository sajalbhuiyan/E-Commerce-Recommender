"""Convert a local download of the Kaggle RecSys 2020 e-commerce dataset into
pickled artifacts for this project.

Usage:
  1. Download and extract the Kaggle dataset to a folder (for example: data/kaggle/)
  2. Run this script pointing at that folder:

     python scripts/convert_kaggle_to_pickles.py --input_dir data/kaggle --out_dir .

The script will attempt to find a products/items CSV and synthesize a
`products.pkl` with columns: product_id, name, brand, price, image_url, category.
If no images are present it seeds stable picsum.photos URLs by product id.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import re
import sys


def find_csv_with_columns(folder: Path, required_any: list):
    """Return the first CSV in folder whose columns include any of the required_any strings."""
    for p in folder.glob('**/*.csv'):
        try:
            df = pd.read_csv(p, nrows=2)
            cols = [c.lower() for c in df.columns]
            for token in required_any:
                if any(token in c for c in cols):
                    return p
        except Exception:
            continue
    return None


def infer_mappings(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    def find_candidate(names):
        for n in names:
            for c in cols:
                if n in c:
                    return cols[c]
        return None

    mapping = {}
    mapping['product_id'] = find_candidate(['product_id', 'productid', 'product', 'item_id', 'itemid', 'sku', 'id'])
    mapping['name'] = find_candidate(['name', 'title', 'product_name', 'productname'])
    mapping['brand'] = find_candidate(['brand', 'manufacturer'])
    mapping['price'] = find_candidate(['price', 'cost', 'retail_price'])
    mapping['image_url'] = find_candidate(['image', 'image_url', 'imageurl', 'picture', 'img'])
    mapping['category'] = find_candidate(['category', 'category_name', 'cat'])
    return mapping


def make_pic_url(pid, w=400, h=400):
    return f'https://picsum.photos/seed/{pid}/{w}/{h}'


def main(input_dir: str, out_dir: str):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    if not input_dir.exists():
        print('Input directory not found:', input_dir)
        sys.exit(1)

    # Try to find a products CSV
    cand = find_csv_with_columns(input_dir, ['product', 'item', 'sku'])
    if not cand:
        print('Could not find a product/items CSV in', input_dir)
        print('Files scanned:', [str(p) for p in input_dir.glob("**/*.csv")])
        sys.exit(1)

    print('Found candidate products CSV:', cand)
    df = pd.read_csv(cand)
    print('Loaded rows:', len(df), 'columns:', list(df.columns))

    mapping = infer_mappings(df)
    print('Inferred column mapping:', mapping)

    # Build output dataframe with desired columns
    out = pd.DataFrame()
    # product id
    pid_col = mapping.get('product_id')
    if pid_col is None:
        # try to create numeric ids
        out['product_id'] = [f'P{i+1000}' for i in range(len(df))]
    else:
        out['product_id'] = df[pid_col].astype(str)

    # name
    name_col = mapping.get('name')
    out['name'] = df[name_col].astype(str) if name_col is not None else out['product_id'].apply(lambda x: f'Product {x}')

    # brand
    brand_col = mapping.get('brand')
    out['brand'] = df[brand_col].astype(str) if brand_col is not None else ''

    # price
    price_col = mapping.get('price')
    if price_col is not None:
        try:
            out['price'] = pd.to_numeric(df[price_col], errors='coerce')
        except Exception:
            out['price'] = None
    else:
        out['price'] = None

    # category
    cat_col = mapping.get('category')
    out['category'] = df[cat_col].astype(str) if cat_col is not None else None

    # image url
    img_col = mapping.get('image_url')
    if img_col is not None:
        out['image_url'] = df[img_col].astype(str)
    else:
        # fallback to picsum seeded by product id
        out['image_url'] = out['product_id'].apply(lambda pid: make_pic_url(pid))

    # Save products.pkl
    out_path = out_dir / 'products.pkl'
    joblib.dump(out, out_path)
    print('Wrote products.pkl to', out_path)
    print(out.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, help='Path to extracted Kaggle dataset folder')
    parser.add_argument('--out_dir', '-o', default='.', help='Output directory to write pickles')
    args = parser.parse_args()
    main(args.input_dir, args.out_dir)
