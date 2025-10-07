"""Simple HTTP API wrapper for the recommender core using FastAPI.

Run locally:
    pip install fastapi uvicorn[standard]
    uvicorn api_server:app --reload --port 8000

Endpoints:
- GET /health
- GET /products?limit=100
- GET /product/{product_id}
- GET /recommend/content?product_id=...&top_n=5
- GET /recommend/hybrid?user_id=...&top_n=5
- GET /recommend/user?user_id=...&top_n=5

This file intentionally keeps dependencies minimal and uses the existing recommender_core module.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import recommender_core as rc
import pandas as pd

app = FastAPI(title="ShopSmart Recommender API")

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def row_to_dict(row: pd.Series):
    """Convert a pandas Series (row) to serializable dict, coerce numpy types."""
    try:
        d = row.to_dict()
    except Exception:
        return {}
    # convert common numpy types
    for k, v in list(d.items()):
        if isinstance(v, (pd.Timestamp,)):
            d[k] = v.isoformat()
        try:
            if hasattr(v, "item"):
                d[k] = v.item()
        except Exception:
            pass
    return d


@app.get("/health")
def health():
    return {
        "ok": True,
        "products_loaded": rc.products_info is not None,
        "embeddings": rc.item_emb is not None,
    }


@app.get("/products")
def list_products(limit: int = Query(100, ge=1, le=1000)):
    if rc.products_info is None:
        return {"products": []}
    df = rc.products_info
    # select common columns safely
    cols = [c for c in ["product_id", "name", "brand", "price", "image_url"] if c in df.columns]
    items = []
    for _, r in df.head(limit).iterrows():
        data = {c: r.get(c) for c in cols}
        # fallback: include product_id if missing in cols
        if "product_id" not in data and "product_id" in df.columns:
            data["product_id"] = r.get("product_id")
        items.append(row_to_dict(pd.Series(data)))
    return {"products": items}


@app.get("/product/{product_id}")
def product_detail(product_id: str):
    if rc.products_info is None:
        raise HTTPException(status_code=404, detail="No product data")
    df = rc.products_info
    found = df[df["product_id"] == product_id]
    if found.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    return row_to_dict(found.iloc[0])


@app.get("/recommend/content")
def recommend_content(product_id: str = Query(..., alias="product_id"), top_n: int = 5):
    try:
        recs = rc.content_recommend(product_id, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # enrich with basic product info when available
    out = []
    for pid in recs:
        item = {"product_id": pid}
        if rc.products_info is not None:
            df = rc.products_info
            found = df[df["product_id"] == pid]
            if not found.empty:
                item.update(row_to_dict(found.iloc[0]))
        out.append(item)
    return {"recommendations": out}


@app.get("/recommend/hybrid")
def recommend_hybrid(user_id: Optional[str] = None, top_n: int = 10):
    try:
        recs = rc.hybrid_recommend(user_id if user_id else None, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out = []
    for pid in recs:
        item = {"product_id": pid}
        if rc.products_info is not None:
            df = rc.products_info
            found = df[df["product_id"] == pid]
            if not found.empty:
                item.update(row_to_dict(found.iloc[0]))
        out.append(item)
    return {"recommendations": out}


@app.get("/recommend/user")
def recommend_user(user_id: str = Query(...), top_n: int = 10):
    try:
        recs = rc.user_recommend(user_id, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out = []
    for pid in recs:
        item = {"product_id": pid}
        if rc.products_info is not None:
            df = rc.products_info
            found = df[df["product_id"] == pid]
            if not found.empty:
                item.update(row_to_dict(found.iloc[0]))
        out.append(item)
    return {"recommendations": out}
