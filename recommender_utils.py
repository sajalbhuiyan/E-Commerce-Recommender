import re
import pandas as pd


def synthesize_title(prod, pid=None):
    """Return a friendly product title. If the dataset contains a generic
    name like "Product 12", synthesize a title using brand/category/price.

    prod may be a dict-like or pandas Series.
    """
    title = None
    try:
        title = prod.get('name') or prod.get('title')
    except Exception:
        # dict-like fallback
        title = prod.get('name') if isinstance(prod, dict) else None

    if title and isinstance(title, str) and re.match(r"^\s*Product\s*\d+\s*$", title, re.IGNORECASE):
        brand = prod.get('brand') or ''
        category = prod.get('category') or ''
        price_val = None
        try:
            if prod.get('price') is not None and not pd.isna(prod.get('price')):
                price_val = float(prod.get('price'))
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
            return " • ".join(parts)
        return f"Product {pid}"

    return title or (f"Product {pid}" if pid is not None else "Product")
