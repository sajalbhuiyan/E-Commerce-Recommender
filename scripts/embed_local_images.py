"""Embed local images into products.pkl as data URIs for local development only.

Usage:
    - Place images in ./images named by product id e.g. P1000.jpg
    - Run: python scripts\embed_local_images.py

This updates products.pkl in-place and writes image data URIs into an 'image_url' column.
"""
from pathlib import Path
import joblib, base64, mimetypes
import pandas as pd
import imghdr

ROOT = Path(__file__).resolve().parent.parent
products_path = ROOT / 'products.pkl'
images_dir = ROOT / 'images'

if not products_path.exists():
    print('products.pkl not found; aborting')
    raise SystemExit(1)

print('Loading products.pkl...')
df = joblib.load(products_path)
if not isinstance(df, pd.DataFrame):
    print('products.pkl is not a DataFrame; aborting')
    raise SystemExit(1)

images_dir.mkdir(exist_ok=True)

def to_data_uri(path: Path):
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0]
    if not mime:
        kind = imghdr.what(None, h=data)
        mime = f'image/{kind}' if kind else 'application/octet-stream'
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:{mime};base64,{b64}'

image_col = []
for _, row in df.iterrows():
    pid = str(row.get('product_id'))
    found = None
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
        p = images_dir / f"{pid}{ext}"
        if p.exists():
            found = p
            break
    if found is None:
        # find any file that contains pid
        for p in images_dir.glob(f"*{pid}*"):
            if p.is_file():
                found = p
                break
    if found:
        try:
            uri = to_data_uri(found)
            image_col.append(uri)
            print(f"Embedded {found.name} for {pid}")
        except Exception as e:
            print('Failed embedding', found, e)
            image_col.append(None)
    else:
        image_col.append(None)

# write to dataframe
df['image_url'] = image_col
joblib.dump(df, products_path)
print('Wrote updated products.pkl with embedded image_url column')
