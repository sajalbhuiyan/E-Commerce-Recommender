from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import difflib
from functools import lru_cache
import json
import os

# Optional Redis caching support. If Redis is not available, cached functions use lru_cache.
try:
    import redis
    _redis_available = True
except Exception:
    redis = None
    _redis_available = False

ROOT = Path(__file__).resolve().parent


def safe_load(path: str):
    full = ROOT / path
    if not full.exists():
        return None
    try:
        return joblib.load(full)
    except Exception as e:
        # we avoid raising so callers can handle missing artifacts
        return None


# load artifacts (callers can re-load if needed)
products_info = safe_load('products.pkl')
item_emb = safe_load('item_emb.pkl')
idx2item = safe_load('idx2item.pkl')
item2idx = safe_load('item2idx.pkl')
X_content = safe_load('X_content.pkl')
tfidf = safe_load('tfidf_vectorizer.pkl')
user_item_mat = safe_load('user_item_mat.pkl')
user2idx = safe_load('user2idx.pkl')
le_user = safe_load('le_user.pkl')
le_item = safe_load('le_item.pkl')


def normalize_idx2item(idx2item_obj):
    if isinstance(idx2item_obj, dict):
        try:
            max_idx = max(int(k) for k in idx2item_obj.keys())
            lst = [None] * (max_idx + 1)
            for k, v in idx2item_obj.items():
                lst[int(k)] = v
            return lst
        except Exception:
            return idx2item_obj
    return idx2item_obj


idx2item = normalize_idx2item(idx2item)


def has_embeddings():
    return item_emb is not None and item2idx is not None and idx2item is not None


def _most_similar_by_embeddings(index: int, top_n: int) -> List:
    emb = np.asarray(item_emb)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    target = emb[index].reshape(1, -1)
    sims = cosine_similarity(target, emb).flatten()
    if index < len(sims):
        sims[index] = -np.inf
    top_idxs = np.argsort(-sims)[:top_n]
    return [idx2item[i] for i in top_idxs]


@lru_cache(maxsize=512)
def similar_indices_for(idx: int, top_n: int = 10):
    """Cached helper to return top indices for an item index using embeddings."""
    if not has_embeddings():
        return []
    emb = np.asarray(item_emb)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    target = emb[idx].reshape(1, -1)
    sims = cosine_similarity(target, emb).flatten()
    if idx < len(sims):
        sims[idx] = -np.inf
    top_idxs = np.argsort(-sims)[:top_n]
    return [int(i) for i in top_idxs]


def content_recommend(product_identifier: str, top_n: int = 5) -> List:
    # direct id
    if item2idx and product_identifier in item2idx:
        idx = item2idx[product_identifier]
        if has_embeddings():
            return _most_similar_by_embeddings(idx, top_n)
        if X_content is not None:
            sims = cosine_similarity(X_content[idx], X_content).flatten()
            top_idxs = np.argsort(-sims)[1:top_n+1]
            return [idx2item[i] for i in top_idxs]

    # search in products_info
    if products_info is not None and isinstance(products_info, pd.DataFrame):
        df = products_info
        mask = pd.Series([False] * len(df))
        if 'product_id' in df.columns:
            mask = mask | (df['product_id'].astype(str) == str(product_identifier))
        if 'name' in df.columns:
            mask = mask | (df['name'].str.contains(str(product_identifier), case=False, na=False))
        if 'brand' in df.columns:
            mask = mask | (df['brand'].str.contains(str(product_identifier), case=False, na=False))
        found = df[mask]
        if not found.empty:
            product_id = found['product_id'].iloc[0]
            if item2idx and product_id in item2idx:
                idx = item2idx[product_id]
                if has_embeddings():
                    return _most_similar_by_embeddings(idx, top_n)
                if X_content is not None:
                    sims = cosine_similarity(X_content[idx], X_content).flatten()
                    top_idxs = np.argsort(-sims)[1:top_n+1]
                    return [idx2item[i] for i in top_idxs]

    return []


def top_popular(n: int = 10) -> List:
    if user_item_mat is None:
        return []
    try:
        # assume user_item_mat is user x item matrix
        pop = np.asarray(user_item_mat).sum(axis=0)
        top = np.argsort(-pop)[:n]
        return [idx2item[int(i)] for i in top]
    except Exception:
        return []


def hybrid_recommend(user_id: Optional[str] = None, top_n: int = 10) -> List:
    # If we have a user-item matrix and a user identifier, try to recommend from that
    if user_item_mat is not None and user_id is not None:
        # map provided user_id to row index
        uidx = None
        try:
            if user2idx and user_id in user2idx:
                uidx = user2idx[user_id]
            elif le_user is not None:
                try:
                    uidx = int(le_user.transform([user_id])[0])
                except Exception:
                    uidx = None
        except Exception:
            uidx = None

        if uidx is not None:
            try:
                row = np.asarray(user_item_mat)[int(uidx)]
                top_idxs = np.argsort(-row)[:top_n]
                return [idx2item[int(i)] for i in top_idxs]
            except Exception:
                pass

        # if we couldn't resolve user, fall back to popular
        return top_popular(top_n)

    # fallback: content around a sample item or by embeddings
    if has_embeddings():
        try:
            return _most_similar_by_embeddings(0, top_n)
        except Exception:
            return []
    if products_info is not None and 'product_id' in products_info.columns:
        return content_recommend(products_info['product_id'].iloc[0], top_n)
    return []


def user_recommend(user_id: str, top_n: int = 10) -> List:
    """Explicit user-based recommendation wrapper (tries collaborative first, then popular)."""
    return hybrid_recommend(user_id=user_id, top_n=top_n)


def create_demo_data(n_items: int = 50, n_users: int = 20):
    """Create small demo artifacts (products.pkl, item_emb.pkl, mappings, user_item_mat.pkl).

    Useful when the repo has no artifacts; writes files to repo root using joblib.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # products dataframe
    ids = [f'P{1000+i}' for i in range(n_items)]
    names = [f'Product {i}' for i in range(n_items)]
    brands = [f'Brand {i%10}' for i in range(n_items)]
    prices = (np.random.rand(n_items) * 100).round(2)
    image_url = ['https://via.placeholder.com/150'] * n_items
    df = pd.DataFrame({'product_id': ids, 'name': names, 'brand': brands, 'price': prices, 'image_url': image_url})

    # embeddings
    emb = np.random.normal(size=(n_items, 32)).astype(np.float32)

    # mappings
    item2idx_local = {pid: i for i, pid in enumerate(ids)}
    idx2item_local = ids

    # users and user_item_mat
    users = [f'U{200+i}' for i in range(n_users)]
    le_u = LabelEncoder().fit(users)
    user2idx_local = {u: int(le_u.transform([u])[0]) for u in users}
    mat = np.random.poisson(0.2, size=(n_users, n_items)).astype(np.float32)

    # dump
    joblib.dump(df, ROOT / 'products.pkl')
    joblib.dump(emb, ROOT / 'item_emb.pkl')
    joblib.dump(item2idx_local, ROOT / 'item2idx.pkl')
    joblib.dump(idx2item_local, ROOT / 'idx2item.pkl')
    joblib.dump(mat, ROOT / 'user_item_mat.pkl')
    joblib.dump(user2idx_local, ROOT / 'user2idx.pkl')
    joblib.dump(le_u, ROOT / 'le_user.pkl')
    # reload local variables into module-level ones so app can pick them up without restart
    global products_info, item_emb, item2idx, idx2item, user_item_mat, user2idx, le_user
    products_info = df
    item_emb = emb
    item2idx = item2idx_local
    idx2item = idx2item_local
    user_item_mat = mat
    user2idx = user2idx_local
    le_user = le_u
    return True


# Cached wrappers to speed repeated calls (keys must be hashable)
@lru_cache(maxsize=1024)
def cached_content_recommend(product_identifier: str, top_n: int = 5) -> List:
    # Try Redis-first caching
    key = f"content:{product_identifier}:{top_n}"
    if _redis_available:
        try:
            val = _redis.get(key)
            if val:
                return tuple(json.loads(val))
        except Exception:
            pass
    res = list(content_recommend(product_identifier, top_n))
    if _redis_available:
        try:
            _redis.set(key, json.dumps(res), ex=3600)
        except Exception:
            pass
    return tuple(res)


@lru_cache(maxsize=1024)
def cached_hybrid_recommend(user_id: Optional[str], top_n: int = 10) -> List:
    # user_id can be None; convert to string for caching
    key_user = str(user_id) if user_id is not None else 'None'
    key = f"hybrid:{key_user}:{top_n}"
    if _redis_available:
        try:
            val = _redis.get(key)
            if val:
                return tuple(json.loads(val))
        except Exception:
            pass
    res = list(hybrid_recommend(user_id if user_id is not None else None, top_n))
    if _redis_available:
        try:
            _redis.set(key, json.dumps(res), ex=3600)
        except Exception:
            pass
    return tuple(res)


@lru_cache(maxsize=1024)
def cached_user_recommend(user_id: str, top_n: int = 10) -> List:
    key = f"user:{user_id}:{top_n}"
    if _redis_available:
        try:
            val = _redis.get(key)
            if val:
                return tuple(json.loads(val))
        except Exception:
            pass
    res = list(user_recommend(user_id, top_n))
    if _redis_available:
        try:
            _redis.set(key, json.dumps(res), ex=3600)
        except Exception:
            pass
    return tuple(res)


_redis = None


def init_redis(host: str = 'localhost', port: int = 6379, db: int = 0, password: str = None):
    """Initialize a module-level Redis client. Returns True on success."""
    global _redis, _redis_available
    if not _redis_available:
        return False
    try:
        client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        # quick ping
        client.ping()
        _redis = client
        return True
    except Exception:
        _redis = None
        return False


def fetch_remote_artifact(url: str, filename: str, checksum: str = None):
    """Download a remote artifact from a URL and save it under ROOT/filename.

    If checksum is provided (sha256 hex), the function will validate it. Returns True on success.
    """
    try:
        import requests, hashlib
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        target = ROOT / filename
        h = hashlib.sha256() if checksum else None
        with open(target, 'wb') as f:
            for chunk in resp.iter_content(1024 * 64):
                if not chunk:
                    continue
                f.write(chunk)
                if h is not None:
                    h.update(chunk)
        if checksum and h is not None:
            if h.hexdigest() != checksum.lower():
                # remove file
                try:
                    target.unlink()
                except Exception:
                    pass
                return False
        # attempt to load into memory if recognized name
        name = filename.lower()
        if 'products' in name:
            global products_info
            products_info = safe_load('products.pkl')
        if 'item_emb' in name:
            global item_emb
            item_emb = safe_load('item_emb.pkl')
        if 'item2idx' in name:
            global item2idx
            item2idx = safe_load('item2idx.pkl')
        if 'idx2item' in name:
            global idx2item
            idx2item = normalize_idx2item(safe_load('idx2item.pkl'))
        if 'user_item_mat' in name:
            global user_item_mat
            user_item_mat = safe_load('user_item_mat.pkl')
        if 'user2idx' in name:
            global user2idx
            user2idx = safe_load('user2idx.pkl')
        return True
    except Exception:
        return False


def fetch_artifacts_from_env():
    """If environment variables or Streamlit secrets provide artifact URLs, download missing files.

    Recognized env vars / secrets:
      - SVD_MODEL_URL
      - RF_MODEL_URL
      - ARTIFACT_SOURCES (JSON mapping filename->url)
    """
    # prefer explicit env vars
    urls = {}
    svd = os.environ.get('SVD_MODEL_URL')
    rf = os.environ.get('RF_MODEL_URL')
    if svd:
        urls['svd_model.pkl'] = svd
    if rf:
        urls['rf_recommender.pkl'] = rf

    # ARTIFACT_SOURCES can be a JSON mapping
    art_src = os.environ.get('ARTIFACT_SOURCES')
    if art_src:
        try:
            mapping = json.loads(art_src)
            for k, v in mapping.items():
                urls[k] = v
        except Exception:
            pass

    # try streamlit secrets if available
    try:
        import streamlit as _st
        secrets = _st.secrets
        if 'SVD_MODEL_URL' in secrets and 'svd_model.pkl' not in urls:
            urls['svd_model.pkl'] = secrets['SVD_MODEL_URL']
        if 'RF_MODEL_URL' in secrets and 'rf_recommender.pkl' not in urls:
            urls['rf_recommender.pkl'] = secrets['RF_MODEL_URL']
        if 'ARTIFACT_SOURCES' in secrets:
            try:
                mapping = json.loads(secrets['ARTIFACT_SOURCES'])
                for k, v in mapping.items():
                    urls[k] = v
            except Exception:
                pass
    except Exception:
        # streamlit not available or no secrets
        pass

    # Download if local file missing
    for fname, url in urls.items():
        target = ROOT / fname
        if not target.exists():
            try:
                fetch_remote_artifact(url, fname)
            except Exception:
                # ignore network errors at import time
                pass


# Note: fetching artifacts from env/secrets is opt-in. Call `fetch_artifacts_from_env()`
# from your Streamlit app after `st.set_page_config(...)` to avoid Streamlit being
# initialized during module import (which would break set_page_config()).


def validate_and_load_model(path: str):
    """Attempt to joblib.load the model and return (True, object) or (False, error_str)."""
    try:
        obj = joblib.load(ROOT / path)
        return True, obj
    except Exception as e:
        return False, str(e)


def convert_sklearn_to_onnx(pkl_path: str, onnx_path: str = None):
    """Try to convert a scikit-learn estimator saved in pkl_path to ONNX.

    Returns tuple (success: bool, message_or_path: str).
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        return False, f'skl2onnx not installed: {e}'

    ok, model_or_err = validate_and_load_model(pkl_path)
    if not ok:
        return False, f'Failed to load model: {model_or_err}'
    model = model_or_err

    # We must guess an input shape. If the model has 'n_features_in_' attribute use it.
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None:
        # best-effort: try 10
        n_features = 10

    initial_type = [('input', FloatTensorType([None, int(n_features)]))]
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        target = ROOT / (onnx_path or (Path(pkl_path).stem + '.onnx'))
        with open(target, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        return True, str(target)
    except Exception as e:
        return False, str(e)


def get_product_choices(limit: int = 100) -> List[str]:
    if products_info is None:
        return []
    df = products_info
    if 'name' in df.columns:
        return df['name'].astype(str).head(limit).tolist()
    if 'brand' in df.columns:
        return df['brand'].astype(str).head(limit).tolist()
    return df['product_id'].astype(str).head(limit).tolist()


def fuzzy_find(query: str, n: int = 10) -> List[str]:
    # Search across product name/brand/product_id using difflib
    if products_info is None:
        return []
    corpus = []
    df = products_info
    if 'name' in df.columns:
        corpus += df['name'].astype(str).tolist()
    if 'brand' in df.columns:
        corpus += df['brand'].astype(str).tolist()
    corpus += df['product_id'].astype(str).tolist()
    # dedupe
    corpus = list(dict.fromkeys(corpus))
    matches = difflib.get_close_matches(query, corpus, n=n, cutoff=0.3)
    return matches


def explain_tfidf(product_identifier: str, top_k: int = 10) -> List[str]:
    if tfidf is None or X_content is None or item2idx is None:
        return []
    # try resolve id
    pid = None
    if product_identifier in item2idx:
        pid = product_identifier
    else:
        if products_info is not None:
            df = products_info
            mask = pd.Series([False] * len(df))
            if 'product_id' in df.columns:
                mask = mask | (df['product_id'].astype(str) == str(product_identifier))
            if 'name' in df.columns:
                mask = mask | (df['name'].str.contains(str(product_identifier), case=False, na=False))
            found = df[mask]
            if not found.empty:
                pid = found['product_id'].iloc[0]
    if pid is None:
        return []
    idx = item2idx[pid]
    try:
        vec = X_content[idx]
        if hasattr(vec, 'toarray'):
            arr = vec.toarray().ravel()
        else:
            arr = np.asarray(vec).ravel()
        top = np.argsort(-arr)[:top_k]
        features = np.array(tfidf.get_feature_names_out())[top]
        return features.tolist()
    except Exception:
        return []
