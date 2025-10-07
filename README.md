# E-Commerce Recommender

This is a lightweight Streamlit app that serves recommendations for an e-commerce dataset.

What this repo expects (optional files that improve results):
- `products.pkl` - a pandas DataFrame with product metadata (columns like `product_id`, `name`, `brand`, `image_url`, `price`)
- `item_emb.pkl` - numpy array or list of item embeddings (shape: n_items x d)
- `item2idx.pkl` and `idx2item.pkl` - mappings between product_id and row index in embeddings
- `X_content.pkl` and `tfidf_vectorizer.pkl` - optional TF-IDF content features
- collaborative models like `svd_model.pkl`, `rf_recommender.pkl` etc. (optional)

How to run

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Start the app:

```powershell
streamlit run app.py
```

Notes

- The app is defensive: missing files are tolerated and the UI will show status in the sidebar.
- For best recommendations provide `item_emb.pkl` and mapping files.
- The code is written to be extended with collaborative or ML-based recommenders.
