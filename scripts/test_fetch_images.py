import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
P = ROOT / 'products.pkl'
if not P.exists():
    print('products.pkl not found at', P)
    sys.exit(1)

df = joblib.load(P)
print('Columns:', list(df.columns))
if 'image_url' not in df.columns:
    print('No image_url column')
    sys.exit(1)

urls = df['image_url'].head(10).tolist()
print('\nSample image_url values:')
for i,u in enumerate(urls, 1):
    print(f'{i}.', u)

print('\nAttempting to fetch first 5 URLs...')
try:
    import requests
except Exception:
    requests = None
    print('requests not available; will use urllib')

from urllib.request import urlopen

for u in urls[:5]:
    if not isinstance(u, str) or not u.startswith('http'):
        print(' - not a http URL:', u)
        continue
    print('\nFetching:', u)
    ok = False
    if requests is not None:
        try:
            r = requests.get(u, timeout=8)
            print(' status_code=', r.status_code, 'content-length=', len(r.content) if r.content is not None else 'None', 'content-type=', r.headers.get('content-type'))
            ok = r.status_code == 200 and len(r.content) > 100
        except Exception as e:
            print(' requests error:', e)
    if not ok:
        try:
            with urlopen(u, timeout=8) as r:
                data = r.read(256)
                print(' urllib read bytes=', len(data), 'peek=', data[:8])
                ok = len(data) > 10
        except Exception as e:
            print(' urllib error:', e)
    print(' reachable=' , ok)

print('\nDone')
