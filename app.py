import streamlit as st
from recommender_core import (
    content_recommend, hybrid_recommend, user_recommend,
    cached_content_recommend, cached_hybrid_recommend, cached_user_recommend,
    get_product_choices, fuzzy_find, explain_tfidf, create_demo_data, fetch_artifacts_from_env
)
import recommender_core as rc
import pandas as pd
import math
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from recommender_utils import synthesize_title
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ShopSmart AI | Advanced Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
        border: 1px solid #f0f0f0;
        position: relative;
        overflow: hidden;
    }
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    .product-image-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
        background: #f8f9fa;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .product-image {
        transition: transform 0.3s ease;
    }
    .product-card:hover .product-image {
        transform: scale(1.05);
    }
    .product-title {
        font-weight: 700;
        font-size: 1.1rem;
        line-height: 1.4;
        color: #2d3748;
        margin-bottom: 0.5rem;
        height: 3rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-price {
        font-size: 1.4rem;
        font-weight: 800;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    .product-original-price {
        font-size: 1rem;
        color: #9e9e9e;
        text-decoration: line-through;
        margin-left: 0.5rem;
    }
    .product-brand {
        color: #667eea;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .product-rating {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .rating-stars {
        color: #ffc107;
        margin-right: 0.5rem;
    }
    .rating-value {
        color: #666;
        font-weight: 600;
    }
    .discount-badge {
        position: absolute;
        top: 15px;
        right: 15px;
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        z-index: 2;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    .recommendation-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .section-header {
        border-left: 5px solid #667eea;
        padding-left: 1.2rem;
        margin: 2.5rem 0 1.5rem 0;
        color: #2d3748;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .trend-up {
        color: #4caf50;
    }
    .trend-down {
        color: #f44336;
    }
    .cart-badge {
        background: #ff6b6b;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        position: absolute;
        top: -5px;
        right: -5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for shopping cart, user session, etc.
if 'cart' not in st.session_state:
    st.session_state.cart = {}
if 'view_history' not in st.session_state:
    st.session_state.view_history = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'price_range': [0, 1000],
        'preferred_categories': [],
        'preferred_brands': []
    }
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = set()
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'show_images' not in st.session_state:
    st.session_state.show_images = True

# Initialize artifacts
try:
    fetch_artifacts_from_env()
except Exception:
    pass

# Header with shopping cart
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<h1 class="main-header">🛍️ ShopSmart AI</h1>', unsafe_allow_html=True)
    st.markdown("### Discover products you'll love with AI-powered recommendations")

with col3:
    cart_count = sum(st.session_state.cart.values())
    cart_text = f"🛒 Cart ({cart_count})" if cart_count > 0 else "🛒 Cart"
    if st.button(cart_text, use_container_width=True):
        st.session_state.show_cart = True

# Enhanced product data functions
def enhance_product_data():
    """Add enhanced product data for demo purposes"""
    if rc.products_info is not None:
        df = rc.products_info
        # Add demo ratings, discounts, etc.
        if 'rating' not in df.columns:
            import numpy as np
            np.random.seed(42)
            df['rating'] = np.round(np.random.uniform(3.5, 5.0, len(df)), 1)
            df['review_count'] = np.random.randint(10, 1000, len(df))
            df['original_price'] = df['price'] * np.random.uniform(1.2, 2.0, len(df))
            df['discount'] = np.where(df['original_price'] > df['price'], 
                                    ((df['original_price'] - df['price']) / df['original_price'] * 100).astype(int), 0)
            df['in_stock'] = np.random.choice([True, False], len(df), p=[0.85, 0.15])
            df['category'] = np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Beauty'], len(df))
            df['shipping_time'] = np.random.choice(['1-2 days', '3-5 days', '1 week'], len(df))
        return df
    return None

def get_product_image(prod):
    """Enhanced image handling with better fallbacks"""
    import base64
    import io
    try:
        import requests
    except Exception:
        requests = None

    img = None
    image_cols = ['image_url', 'image', 'img', 'image_link', 'thumbnail', 'imageURL', 'product_image']
    
    for col in image_cols:
        if col in prod.index:
            img = prod.get(col)
            if img and not pd.isna(img):
                break
    
    # Handle list/tuple images
    if img is not None:
        try:
            if isinstance(img, (list, tuple)):
                img = img[0] if len(img) > 0 else None
            elif hasattr(img, 'shape') and len(img.shape) > 0:
                img = img[0] if img.size > 0 else None
        except Exception:
            pass
    
    # Better placeholder based on product category
    if img is None or (isinstance(img, str) and img.strip() == ''):
        category = prod.get('category', 'product')
        category_colors = {
            'electronics': '4CAF50', 'clothing': '2196F3', 'home': 'FF9800', 
            'books': '795548', 'beauty': 'E91E63', 'sports': 'FF5722'
        }
        color = category_colors.get(str(category).lower(), '667eea')
        img = f'https://via.placeholder.com/400x400/{color}/ffffff?text={category.title().replace(" ", "+")}'
        # If we have a product name/brand, try to get a relevant image from Unsplash
        try:
            from urllib.parse import quote_plus
            q_parts = []
            name = prod.get('name') or prod.get('title')
            if name and isinstance(name, str):
                q_parts.append(name)
            brand = prod.get('brand')
            if brand and not pd.isna(brand):
                q_parts.append(str(brand))
            if category and not pd.isna(category):
                q_parts.append(str(category))
            if q_parts:
                query = quote_plus(' '.join(q_parts))
                # Unsplash source with query returns a relevant image
                img = f'https://source.unsplash.com/400x400/?{query}'
        except Exception:
            pass

    # If img is a remote URL, try to fetch and cache a base64 data URI in session state
    try:
        if isinstance(img, str) and img.startswith('http'):
            cache = st.session_state.get('_image_cache', {})
            if img in cache:
                return cache[img]

            # Try to fetch image bytes
            img_bytes = None
            if requests is not None:
                try:
                    resp = requests.get(img, timeout=6)
                    resp.raise_for_status()
                    img_bytes = resp.content
                except Exception:
                    img_bytes = None
            else:
                # fallback to urllib
                try:
                    from urllib.request import urlopen
                    with urlopen(img, timeout=6) as r:
                        img_bytes = r.read()
                except Exception:
                    img_bytes = None

            if img_bytes:
                try:
                    b64 = base64.b64encode(img_bytes).decode('ascii')
                    # Try to detect mime type from URL extension
                    mime = 'image/jpeg'
                    if img.lower().endswith('.png'):
                        mime = 'image/png'
                    data_uri = f'data:{mime};base64,{b64}'
                    cache[img] = data_uri
                    st.session_state['_image_cache'] = cache
                    return data_uri
                except Exception:
                    pass

            # If fetching failed, return the original URL (Streamlit may still load it)
            return img
    except Exception:
        # any error, fall back to returning whatever we have
        return img
    
    return img

def create_product_card(prod, pid, show_badge=False, badge_text="Recommended", context="general"):
    """Create a professional product card with enhanced features"""
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        
        # Discount badge
        discount = prod.get('discount', 0)
        if discount > 0:
            st.markdown(f'<div class="discount-badge">-{int(discount)}%</div>', unsafe_allow_html=True)
        
        # Recommendation badge
        if show_badge:
            st.markdown(f'<div class="recommendation-badge">{badge_text}</div>', unsafe_allow_html=True)
        
        # Product image container (optional)
        st.markdown('<div class="product-image-container">', unsafe_allow_html=True)
        if st.session_state.get('show_images', True):
            img_url = get_product_image(prod)
            try:
                # Use a fixed width so the layout remains stable across cards
                st.image(img_url, use_column_width=False, width=300, output_format='auto')
            except Exception:
                st.image('https://via.placeholder.com/400x400/cccccc/969696?text=Image+Not+Found', 
                        use_column_width=False, width=300)
        else:
            # Reserve space so the card layout doesn't collapse when images are hidden
            st.markdown('<div style="height:220px"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Product info
        # Use synthesized title when original is a generic placeholder.
        orig_name = (prod.get('name') or prod.get('title'))
        synth_title = synthesize_title(prod, pid)
        # Display synthesized title prominently
        st.markdown(f'<div class="product-title">{synth_title}</div>', unsafe_allow_html=True)
        # If the original name exists and is different, show it muted underneath
        if orig_name and isinstance(orig_name, str) and orig_name.strip() and orig_name.strip().lower() != synth_title.strip().lower():
            st.markdown(f'<div style="color:#8a8f98;font-size:0.9rem;margin-top:0.25rem">{orig_name}</div>', unsafe_allow_html=True)
        # Keep `title` variable for downstream use (e.g., view history)
        title = synth_title
        
        # Brand
        brand = prod.get('brand')
        if brand and not pd.isna(brand):
            st.markdown(f'<div class="product-brand">{brand}</div>', unsafe_allow_html=True)
        
        # Rating
        rating = prod.get('rating')
        if rating and not pd.isna(rating):
            review_count = prod.get('review_count', 0)
            stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
            st.markdown(f'<div class="product-rating"><span class="rating-stars">{stars}</span><span class="rating-value">{rating} ({review_count})</span></div>', unsafe_allow_html=True)
        
        # Price
        price = prod.get('price')
        original_price = prod.get('original_price')
        if price is not None and not pd.isna(price):
            try:
                price_val = float(price)
                if original_price and original_price > price_val:
                    st.markdown(f'<div class="product-price">${price_val:.2f}<span class="product-original-price">${original_price:.2f}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="product-price">${price_val:.2f}</div>', unsafe_allow_html=True)
            except (ValueError, TypeError):
                st.markdown(f'<div class="product-price">{price}</div>', unsafe_allow_html=True)
        
        # Stock status
        in_stock = prod.get('in_stock', True)
        stock_text = "✅ In Stock" if in_stock else "❌ Out of Stock"
        st.caption(stock_text)
        
        # Action buttons: try to render in a row, but fall back to stacked buttons if nested columns
        try:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                cart_count = st.session_state.cart.get(pid, 0)
                if st.button("🛒", key=f"cart_{pid}_{context}", help="Add to cart", use_container_width=True):
                    st.session_state.cart[pid] = cart_count + 1
                    st.experimental_rerun()

            with col2:
                is_in_wishlist = pid in st.session_state.wishlist
                wishlist_icon = "❤️" if is_in_wishlist else "🤍"
                if st.button(wishlist_icon, key=f"wish_{pid}_{context}", help="Add to wishlist", use_container_width=True):
                    if is_in_wishlist:
                        st.session_state.wishlist.remove(pid)
                    else:
                        st.session_state.wishlist.add(pid)
                    st.experimental_rerun()

            with col3:
                if st.button("👁️", key=f"view_{pid}_{context}", help="View details", use_container_width=True):
                    st.session_state.view_history.append({
                        'product_id': pid,
                        'product_name': title,
                        'timestamp': datetime.now(),
                        'context': context
                    })
                    st.session_state.current_product_view = pid
                    st.experimental_rerun()
        except Exception:
            # Fallback: stacked buttons (works inside nested layouts)
            cart_count = st.session_state.cart.get(pid, 0)
            if st.button("🛒 Add to cart", key=f"cart_fallback_{pid}_{context}"):
                st.session_state.cart[pid] = cart_count + 1
                st.experimental_rerun()

            is_in_wishlist = pid in st.session_state.wishlist
            wishlist_icon = "Remove from wishlist" if is_in_wishlist else "Add to wishlist"
            if st.button(wishlist_icon, key=f"wish_fallback_{pid}_{context}"):
                if is_in_wishlist:
                    st.session_state.wishlist.remove(pid)
                else:
                    st.session_state.wishlist.add(pid)
                st.experimental_rerun()

            if st.button("View details", key=f"view_fallback_{pid}_{context}"):
                st.session_state.view_history.append({
                    'product_id': pid,
                    'product_name': title,
                    'timestamp': datetime.now(),
                    'context': context
                })
                st.session_state.current_product_view = pid
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/667eea/ffffff?text=ShopSmart+AI", use_column_width=True)
    st.markdown("---")
    
    # User profile section
    st.markdown(f"### 👤 Welcome, {st.session_state.user_id}!")
    
    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cart Items", sum(st.session_state.cart.values()))
    with col2:
        st.metric("Wishlist", len(st.session_state.wishlist))
    
    st.markdown("---")
    st.markdown("### 🎯 Recommendation Settings")
    
    algorithm = st.selectbox("Algorithm:", ["Content-Based", "Hybrid", "User-Based", "Collaborative Filtering"])
    top_n = st.slider("Number of Recommendations:", 5, 25, 12)
    
    # Advanced filters
    st.markdown("### 🔧 Filters")
    price_range = st.slider("Price Range ($)", 0, 2000, (0, 500), key="price_filter")
    
    if rc.products_info is not None:
        categories = rc.products_info.get('category', pd.Series(['All'])).unique()
        selected_categories = st.multiselect("Categories", options=['All'] + list(categories), default=['All'])
    
    sort_by = st.selectbox("Sort by", ["Relevance", "Price: Low to High", "Price: High to Low", "Rating", "Most Popular"])
    
    st.markdown("---")
    
    # Enhanced product search
    st.markdown("### 🔍 Product Search")
    
    def build_display_choices(limit=500):
        choices = []
        mapping = {}
        if rc.products_info is None:
            return choices, mapping
        
        df = rc.products_info
        cols = df.columns
        
        for _, row in df.head(limit).iterrows():
            pid = row.get('product_id')
            name = row.get('name') if 'name' in cols else None
            brand = row.get('brand') if 'brand' in cols else None
            price = row.get('price') if 'price' in cols else None
            category = row.get('category') if 'category' in cols else None
            
            label_parts = []
            if name and not pd.isna(name):
                label_parts.append(str(name))
            if brand and not pd.isna(brand):
                label_parts.append(f"by {brand}")
            if price is not None and not pd.isna(price):
                try:
                    label_parts.append(f"${float(price):.2f}")
                except Exception:
                    pass
            
            display = " • ".join(label_parts) if label_parts else (rc.products_info[rc.products_info['product_id']==pid]['name'].iloc[0] if (rc.products_info is not None and 'product_id' in rc.products_info.columns and not rc.products_info[rc.products_info['product_id']==pid].empty and 'name' in rc.products_info.columns) else f"Product {pid}")
            full_display = f"{display} (#{pid})"
            mapping[full_display] = pid
            choices.append(full_display)
        
        return choices, mapping

    choices, choices_map = build_display_choices(500)
    
    if choices:
        product_choice = st.selectbox("Select a product:", choices, index=0)
        query = choices_map.get(product_choice, product_choice.split("#")[-1].rstrip(")"))
    else:
        query = st.text_input("Enter Product ID or Name:")
    
    # Recent searches
    if st.session_state.recent_searches:
        st.markdown("**Recent Searches:**")
        for search in st.session_state.recent_searches[-5:]:
            if st.button(f"🔍 {search}", key=f"recent_{search}"):
                query = search
                st.experimental_rerun()

def get_personalized_recommendations(user_id, n=12):
    """Return personalized recommendations combining collaborative, content and popularity.
    This helper is defined before the UI so tabs can call it safely.
    """
    recs = []
    # Try collaborative / user-based first
    try:
        recs = list(cached_user_recommend(user_id, n))
    except Exception:
        recs = []

    # If not enough results, augment from recently viewed products (content-based)
    if len(recs) < n and st.session_state.get('view_history'):
        for v in reversed(st.session_state.view_history[-5:]):
            pid = v.get('product_id')
            if not pid:
                continue
            try:
                more = list(cached_content_recommend(str(pid), n))
            except Exception:
                more = []
            for m in more:
                if m not in recs:
                    recs.append(m)
                if len(recs) >= n:
                    break
            if len(recs) >= n:
                break

    # Fill with popular items if still short
    if len(recs) < n:
        try:
            pop = list(rc.top_popular(n * 2))
        except Exception:
            pop = []
        for p in pop:
            if p not in recs:
                recs.append(p)
            if len(recs) >= n:
                break

    return recs[:n]

# Main tabs with enhanced functionality
def get_trending_products(n=8):
    """Get trending products based on views and cart additions"""
    # This would integrate with actual analytics in a real system
    if rc.products_info is not None:
        return rc.products_info.sample(min(n, len(rc.products_info)))
    return []


def prefetch_urls(urls):
    """Prefetch a list of URLs into session _image_cache as data URIs."""
    import base64
    try:
        import requests
    except Exception:
        requests = None
    cache = st.session_state.get('_image_cache', {})
    count = 0
    for url in urls:
        if not url or not isinstance(url, str) or not url.startswith('http'):
            continue
        if url in cache:
            count += 1
            continue
        data = None
        if requests is not None:
            try:
                r = requests.get(url, timeout=6)
                if r.status_code == 200 and r.content:
                    data = r.content
            except Exception:
                data = None
        if data is None:
            try:
                from urllib.request import urlopen
                with urlopen(url, timeout=6) as r:
                    data = r.read()
            except Exception:
                data = None
        if data:
            try:
                b64 = base64.b64encode(data).decode('ascii')
                mime = 'image/jpeg'
                if url.lower().endswith('.png'):
                    mime = 'image/png'
                cache[url] = f'data:{mime};base64,{b64}'
                count += 1
            except Exception:
                pass
    st.session_state['_image_cache'] = cache
    return count

# Main tabs with enhanced functionality
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard", 
    "🎯 Discover", 
    "🤖 Smart Recs", 
    "👤 Personal", 
    "🛒 Cart & Orders",
    "📊 Analytics"
])

# Tab 1: Dashboard
with tab1:
    st.markdown('<h2 class="section-header">Shopping Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(rc.products_info) if rc.products_info is not None else 0
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{total_products}</div>
            <div class="metric-label">Total Products</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.cart)}</div>
            <div class="metric-label">Cart Items</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.wishlist)}</div>
            <div class="metric-label">Wishlist</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        viewed_count = len(st.session_state.view_history)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{viewed_count}</div>
            <div class="metric-label">Products Viewed</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Recent activity and recommendations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔥 Trending Products")
        # Display trending products (demo)
        if rc.products_info is not None:
            trending_products = rc.products_info.sample(min(8, len(rc.products_info)))
            # Pre-fetch images for the trending products so the UI shows images immediately
            try:
                img_urls = []
                for _, r in trending_products.iterrows():
                    url = r.get('image_url')
                    if url:
                        img_urls.append(url)
                if img_urls:
                    prefetch_urls(img_urls)
            except Exception:
                pass
            cols = st.columns(4)
            for i, (_, product) in enumerate(trending_products.iterrows()):
                with cols[i % 4]:
                    create_product_card(product, product['product_id'], show_badge=True, badge_text="Trending")
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        # View history chart
        if st.session_state.view_history:
            recent_views = pd.DataFrame(st.session_state.view_history[-10:])
            st.dataframe(recent_views[['product_name', 'timestamp']].tail(5), use_container_width=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Recommendations", use_container_width=True):
                st.experimental_rerun()
        
        if st.button("📋 View Order History", use_container_width=True):
            st.session_state.show_orders = True
        
        if st.button("❤️ View Wishlist", use_container_width=True):
            st.session_state.show_wishlist = True

# Tab 2: Discover (Enhanced content-based recommendations)
with tab2:
    st.markdown('<h2 class="section-header">Discover Similar Products</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"Finding products similar to: **{query}**")
        
        # Product details of searched item
        if rc.products_info is not None:
            searched_product = rc.products_info[rc.products_info['product_id'] == query]
            if not searched_product.empty:
                prod = searched_product.iloc[0]
                st.markdown("**Current Product:**")
                create_product_card(prod, query, show_badge=False, context="search")
    
    with col3:
        if st.button("🎯 Find Similar", use_container_width=True, type="primary"):
            if query not in st.session_state.recent_searches:
                st.session_state.recent_searches.append(query)
            st.session_state.run_content_recs = True
    
    if st.session_state.get('run_content_recs', False):
        with st.spinner("🔍 Analyzing product features and finding perfect matches..."):
            recs = cached_content_recommend(str(query), top_n)
            recs_list = list(recs)
        
        if not recs_list:
            st.error("""
            No recommendations found. This could be because:
            - The product ID doesn't exist in our database
            - Similar products are not available
            - Try a different product or check the product database
            """)
        else:
            st.success(f"🎉 Found {len(recs_list)} similar products!")
            
            # Similarity scores and explanations
            with st.expander("📊 Why these recommendations?"):
                st.write("These products share similar features with your selected item:")
                features = explain_tfidf(query, top_k=8)
                if features:
                    st.write("**Key matching features:**", ", ".join(features))
                
                # Similarity distribution
                # Show similarity scores with product names when possible
                similarity_scores = []
                for i in range(min(5, len(recs_list))):
                    label_pid = recs_list[i]
                    label_name = None
                    if rc.products_info is not None and 'product_id' in rc.products_info.columns and 'name' in rc.products_info.columns:
                        found = rc.products_info[rc.products_info['product_id'] == label_pid]
                        if not found.empty:
                            label_name = found.iloc[0].get('name')
                    label_display = label_name or f"Product {i+1}"
                    similarity_scores.append(f"{label_display}: {(len(recs_list)-i)/len(recs_list)*100:.1f}%")
                st.write("**Similarity scores:**", " | ".join(similarity_scores))
            
            # Display recommendations with pagination
            per_page = 8
            total_pages = max(1, math.ceil(len(recs_list) / per_page))
            page = st.session_state.get('content_page', 1)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_recs = recs_list[start_idx:end_idx]
            
            # Product grid
            st.markdown(f"### 📦 Similar Products (Page {page}/{total_pages})")
            cols = st.columns(4)
            for i, pid in enumerate(page_recs):
                with cols[i % 4]:
                    if rc.products_info is not None:
                        product_data = rc.products_info[rc.products_info['product_id'] == pid]
                        if not product_data.empty:
                            similarity_score = f"{(len(recs_list)-i)/len(recs_list)*100:.0f}% Match"
                            create_product_card(product_data.iloc[0], pid, show_badge=True, 
                                              badge_text=similarity_score, context="content_recs")
            
            # Enhanced pagination
            if total_pages > 1:
                st.markdown("---")
                pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 2, 1, 1])
                with pag_col1:
                    if page > 1 and st.button("⬅️ Previous", key="prev_content"):
                        st.session_state.content_page = page - 1
                        st.experimental_rerun()
                with pag_col2:
                    st.write(f"**Page {page} of {total_pages}** - Showing {len(page_recs)} products")
                with pag_col3:
                    if page < total_pages and st.button("Next ➡️", key="next_content"):
                        st.session_state.content_page = page + 1
                        st.experimental_rerun()
                with pag_col4:
                    if st.button("🔄 Refresh", key="refresh_content"):
                        st.session_state.content_page = 1
                        st.experimental_rerun()

# Tab 3: Smart Recs (Hybrid / Personalized / Content)
with tab3:
    st.markdown('<h2 class="section-header">🤖 Smart Recommendations</h2>', unsafe_allow_html=True)
    col_main, col_side = st.columns([3, 1])
    with col_side:
        rec_type = st.radio("Recommendation type", ["For You (Hybrid)", "By Product (Content)", "Top Popular", "Trending"], index=0)
        top_n = st.slider("Results", min_value=4, max_value=24, value=12, step=4)
        if rec_type == "By Product (Content)":
            prod_choice = st.selectbox("Seed product", choices if choices else get_product_choices(100))
            seed_pid = choices_map.get(prod_choice, prod_choice.split("#")[-1].rstrip(")")) if choices else prod_choice
        else:
            seed_pid = None

    # compute recommendations
    recs = []
    if rec_type == "For You (Hybrid)":
        recs = list(cached_hybrid_recommend(st.session_state.user_id, top_n))
    elif rec_type == "By Product (Content)":
        if seed_pid:
            recs = list(cached_content_recommend(str(seed_pid), top_n))
    elif rec_type == "Top Popular":
        recs = list(rc.top_popular(top_n))
    elif rec_type == "Trending":
        recs = list(get_trending_products(top_n))

    if not recs:
        st.info("No recommendations available. Try a different option or initialize demo data.")
    else:
        st.markdown(f"### ✨ {len(recs)} Recommendations")
        per_page = 8
        page = st.session_state.get('smart_page', 1)
        total_pages = max(1, math.ceil(len(recs) / per_page))
        start = (page - 1) * per_page
        page_items = recs[start:start+per_page]

        cols = st.columns(4)
        for i, pid in enumerate(page_items):
            with cols[i % 4]:
                if isinstance(pid, (list, tuple)):
                    # some functions return product rows
                    prod = pid[0] if pid else None
                    if prod is not None and hasattr(prod, 'get'):
                        create_product_card(prod, prod.get('product_id', f'P{i}'), show_badge=True, badge_text='Recommended', context='smart')
                        continue
                # pid may be a product id
                if rc.products_info is not None and 'product_id' in rc.products_info.columns:
                    dfp = rc.products_info[rc.products_info['product_id'] == pid]
                    if not dfp.empty:
                        create_product_card(dfp.iloc[0], pid, show_badge=True, badge_text='Recommended', context='smart')
        # pagination controls
        if total_pages > 1:
            pag1, pag2, pag3 = st.columns([1, 2, 1])
            with pag1:
                if page > 1 and st.button("⬅️ Prev", key='smart_prev'):
                    st.session_state.smart_page = page - 1
                    st.experimental_rerun()
            with pag2:
                st.write(f"Page {page} of {total_pages}")
            with pag3:
                if page < total_pages and st.button("Next ➡️", key='smart_next'):
                    st.session_state.smart_page = page + 1
                    st.experimental_rerun()

# Tab 4: Personal (profile, view history, wishlist)
with tab4:
    st.markdown('<h2 class="section-header">👤 Personal</h2>', unsafe_allow_html=True)
    st.write(f"**User:** {st.session_state.user_id}")
    colp1, colp2 = st.columns([2, 1])
    with colp1:
        st.markdown("### 🧾 View History")
        if st.session_state.view_history:
            vh = pd.DataFrame(st.session_state.view_history)
            st.dataframe(vh[['product_name', 'timestamp', 'context']].tail(20), use_container_width=True)
            if st.button("Clear History"):
                st.session_state.view_history = []
                st.experimental_rerun()
        else:
            st.info("No view history yet. View some products to seed personalized recommendations.")

        st.markdown("### 💡 Recommendations For You")
        personal_recs = list(get_personalized_recommendations(st.session_state.user_id, n=12))
        if personal_recs:
            cols = st.columns(4)
            for i, pid in enumerate(personal_recs[:12]):
                with cols[i % 4]:
                    if rc.products_info is not None:
                        dfp = rc.products_info[rc.products_info['product_id'] == pid]
                        if not dfp.empty:
                            create_product_card(dfp.iloc[0], pid, show_badge=True, badge_text='For You', context='personal')
        else:
            st.info('No personalized recommendations yet. Interact with products or initialize demo data.')

    with colp2:
        st.markdown('### ❤️ Wishlist')
        if st.session_state.wishlist:
            for pid in list(st.session_state.wishlist):
                if rc.products_info is not None:
                    dfp = rc.products_info[rc.products_info['product_id'] == pid]
                    title = dfp.iloc[0].get('name') if not dfp.empty else pid
                else:
                    title = pid
                st.write(f"• {title} ")
            if st.button('Clear Wishlist'):
                st.session_state.wishlist = set()
                st.experimental_rerun()
        else:
            st.info('Your wishlist is empty. Click the ❤ on product cards to add items.')

# Tab 5: Cart & Orders (full page)
with tab5:
    st.markdown('<h2 class="section-header">🛒 Cart & Orders</h2>', unsafe_allow_html=True)
    # Cart section
    st.markdown('## Your Cart')
    if not st.session_state.cart:
        st.info('Your cart is empty. Add items from product listings.')
    else:
        total_amount = 0.0
        for pid, qty in list(st.session_state.cart.items()):
            if rc.products_info is not None:
                dfp = rc.products_info[rc.products_info['product_id'] == pid]
                if not dfp.empty:
                    prod = dfp.iloc[0]
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"**{prod.get('name', pid)}**")
                        st.write(f"{prod.get('brand', '')} - {prod.get('category', '')}")
                    with col2:
                        st.write(f"Qty: {qty}")
                    with col3:
                        new_qty = st.number_input('', min_value=0, max_value=20, value=qty, key=f'cart_qty_{pid}')
                        if new_qty != qty:
                            if new_qty == 0:
                                del st.session_state.cart[pid]
                            else:
                                st.session_state.cart[pid] = new_qty
                            st.experimental_rerun()
                    with col4:
                        line_total = prod.get('price', 0) * qty
                        total_amount += line_total
                        st.write(f"${line_total:.2f}")

        st.markdown(f"### Total: ${total_amount:.2f}")
        colc1, colc2 = st.columns(2)
        with colc1:
            if st.button('← Continue Shopping'):
                # just navigate back to dashboard
                st.session_state.show_cart = False
                st.experimental_rerun()
        with colc2:
            if st.button('🚀 Checkout'):
                # create a simple order record
                order = {
                    'order_id': f"O{int(time.time())}",
                    'items': dict(st.session_state.cart),
                    'total': total_amount,
                    'timestamp': datetime.now()
                }
                st.session_state.orders.append(order)
                st.success('Order placed!')
                st.session_state.cart = {}
                st.experimental_rerun()

    # Orders history
    st.markdown('## Your Orders')
    if st.session_state.orders:
        for o in reversed(st.session_state.orders):
            with st.expander(f"Order {o['order_id']} - ${o['total']:.2f} - {o['timestamp']:%Y-%m-%d %H:%M:%S}"):
                for pid, qty in o['items'].items():
                    if rc.products_info is not None:
                        dfp = rc.products_info[rc.products_info['product_id'] == pid]
                        name = dfp.iloc[0].get('name') if not dfp.empty else pid
                    else:
                        name = pid
                    st.write(f"• {name} x {qty}")
    else:
        st.info('No orders yet.')

# Tab 6: Analytics
with tab6:
    st.markdown('<h2 class="section-header">Shopping Analytics</h2>', unsafe_allow_html=True)
    
    if rc.products_info is not None:
        # Price distribution
        st.markdown("### 💰 Price Distribution")
        prices = rc.products_info['price'].dropna()
        fig = px.histogram(prices, nbins=20, title="Product Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        if 'category' in rc.products_info.columns:
            st.markdown("### 🏷️ Category Distribution")
            category_counts = rc.products_info['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    # User behavior analytics
    st.markdown("### 👤 Your Shopping Behavior")
    if st.session_state.view_history:
        view_df = pd.DataFrame(st.session_state.view_history)
        view_df['hour'] = view_df['timestamp'].dt.hour
        hourly_views = view_df['hour'].value_counts().sort_index()
        
        fig = px.bar(x=hourly_views.index, y=hourly_views.values, 
                     title="Your Product Viewing Pattern by Hour")
        st.plotly_chart(fig, use_container_width=True)

# Shopping Cart Modal
if st.session_state.get('show_cart', False):
    with st.container():
        st.markdown("## 🛒 Shopping Cart")
        
        if not st.session_state.cart:
            st.info("Your cart is empty. Start shopping!")
        else:
            total_amount = 0
            for pid, quantity in st.session_state.cart.items():
                if rc.products_info is not None:
                    product_data = rc.products_info[rc.products_info['product_id'] == pid]
                    if not product_data.empty:
                        prod = product_data.iloc[0]
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.write(f"**{prod.get('name', pid)}**")
                            st.write(f"Price: ${prod.get('price', 0):.2f}")
                        with col2:
                            st.write(f"Qty: {quantity}")
                        with col3:
                            new_qty = st.number_input("", min_value=0, max_value=10, value=quantity, key=f"qty_{pid}")
                            if new_qty != quantity:
                                st.session_state.cart[pid] = new_qty
                                if new_qty == 0:
                                    del st.session_state.cart[pid]
                                st.experimental_rerun()
                        with col4:
                            item_total = prod.get('price', 0) * quantity
                            total_amount += item_total
                            st.write(f"${item_total:.2f}")
            
            st.markdown(f"### Total: ${total_amount:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Continue Shopping", use_container_width=True):
                    st.session_state.show_cart = False
                    st.experimental_rerun()
            with col2:
                if st.button("🚀 Checkout", type="primary", use_container_width=True):
                    st.success("Order placed successfully! 🎉")
                    st.session_state.cart = {}
                    st.session_state.show_cart = False
                    st.experimental_rerun()

# Demo data initialization
if st.sidebar.button("🔄 Initialize Demo Data", use_container_width=True):
    with st.spinner("Creating rich demo dataset..."):
        ok = create_demo_data(n_items=200, n_users=100)
        if ok:
            st.sidebar.success("Demo data created! Refresh to see enhanced features.")
            st.experimental_rerun()
        else:
            st.sidebar.error("Failed to create demo data.")

# System status in sidebar
with st.sidebar.expander("📊 System Status"):
    st.write("**Data Status:**")
    st.write(f"• Products: {'✅ Loaded' if rc.products_info is not None else '❌ Missing'}")
    st.write(f"• Embeddings: {'✅ Ready' if rc.item_emb is not None else '❌ Missing'}")
    st.write(f"• Recommendations: {'✅ Active' if rc.products_info is not None else '❌ Inactive'}")
    
    st.write("**Session Info:**")
    st.write(f"• User: {st.session_state.user_id}")
    st.write(f"• Session started: {datetime.now().strftime('%H:%M:%S')}")

# Prefetch toggle: load first N images into session cache as data URIs
def prefetch_images(n=12):
    import base64
    try:
        import requests
    except Exception:
        requests = None
    cache = st.session_state.get('_image_cache', {})
    if rc.products_info is None or 'image_url' not in rc.products_info.columns:
        return 0
    count = 0
    for _, row in rc.products_info.head(n).iterrows():
        url = row.get('image_url')
        if not url or not isinstance(url, str) or not url.startswith('http'):
            continue
        if url in cache:
            count += 1
            continue
        data = None
        if requests is not None:
            try:
                r = requests.get(url, timeout=6)
                if r.status_code == 200 and r.content:
                    data = r.content
            except Exception:
                data = None
        if data is None:
            try:
                from urllib.request import urlopen
                with urlopen(url, timeout=6) as r:
                    data = r.read()
            except Exception:
                data = None
        if data:
            try:
                b64 = base64.b64encode(data).decode('ascii')
                mime = 'image/jpeg'
                if url.lower().endswith('.png'):
                    mime = 'image/png'
                cache[url] = f'data:{mime};base64,{b64}'
                count += 1
            except Exception:
                pass
    st.session_state['_image_cache'] = cache
    return count

with st.sidebar:
    st.markdown('---')
    prefetch_enable = st.checkbox('Prefetch images (embed as data URIs)', value=False)
    if prefetch_enable:
        n = st.number_input('Prefetch first N images', min_value=4, max_value=100, value=12, step=4)
        with st.spinner('Prefetching images...'):
            got = prefetch_images(n)
            st.success(f'Prefetched {got} images into session cache')
    # Allow user to upload a CSV mapping product_id -> image_url (useful to supply Amazon image URLs)
    st.markdown('---')
    st.markdown('Upload CSV with columns: product_id,image_url')
    csv_file = st.file_uploader('Image mapping CSV', type=['csv'])
    if csv_file is not None:
        try:
            import pandas as _pd
            mapping = _pd.read_csv(csv_file)
            if 'product_id' in mapping.columns and 'image_url' in mapping.columns:
                # merge into products_info and persist
                if rc.products_info is None:
                    st.error('No products_info loaded yet; cannot apply mapping')
                else:
                    df_merge = rc.products_info.merge(mapping[['product_id','image_url']], on='product_id', how='left', suffixes=('','_new'))
                    # prefer new urls when present
                    df_merge['image_url'] = df_merge['image_url_new'].combine_first(df_merge.get('image_url'))
                    df_merge = df_merge.drop(columns=[c for c in df_merge.columns if c.endswith('_new')])
                    import joblib as _joblib
                    _joblib.dump(df_merge, Path(__file__).resolve().parent / 'products.pkl')
                    # reload into module state
                    rc.products_info = df_merge
                    st.success('Applied image mapping and updated products.pkl')
            else:
                st.error('CSV must contain product_id and image_url columns')
        except Exception as e:
            st.error(f'Failed to apply mapping: {e}')

    st.markdown('---')
    st.markdown('Amazon Product Advertising API (optional)')
    pa_access = st.text_input('PA-API Access Key', type='password')
    pa_secret = st.text_input('PA-API Secret Key', type='password')
    pa_tag = st.text_input('PA-API Partner Tag (Associate Tag)')
    pa_region = st.selectbox('PA-API Region', ['us-east-1', 'eu-west-1', 'us-west-2'], index=0)
    pa_host = st.text_input('PA-API Host', value='webservices.amazon.com')
    if st.button('Fetch latest products from Amazon by keyword'):
        kw = st.text_input('Search keyword', value='headphones')
        if not (pa_access and pa_secret and pa_tag):
            st.error('Please provide Access Key, Secret Key and Partner Tag')
        else:
            try:
                import amazon_paapi as _pa
                items = _pa.search_items(kw, pa_access, pa_secret, pa_tag, region=pa_region, host=pa_host, page=1, item_count=10)
                # map items into products_info rows
                new_rows = []
                for it in items:
                    asin = it.get('ASIN') or it.get('asin')
                    title = None
                    img = None
                    brand = None
                    price = None
                    try:
                        title = it.get('ItemInfo', {}).get('Title', {}).get('DisplayValue')
                    except Exception:
                        title = None
                    try:
                        img = it.get('Images', {}).get('Primary', {}).get('Medium', {}).get('URL')
                    except Exception:
                        img = None
                    try:
                        bl = it.get('ItemInfo', {}).get('ByLineInfo', {})
                        brand = bl.get('Brand', {}).get('DisplayValue')
                    except Exception:
                        brand = None
                    try:
                        offers = it.get('Offers', {}).get('Listings', [])
                        if offers:
                            price = offers[0].get('Price', {}).get('Amount')
                    except Exception:
                        price = None
                    if asin:
                        new_rows.append({'product_id': asin, 'name': title or asin, 'brand': brand or '', 'price': price, 'image_url': img})
                if new_rows:
                    import pandas as _pd, joblib as _joblib
                    newdf = _pd.DataFrame(new_rows)
                    # merge with existing products_info
                    if rc.products_info is None:
                        rc.products_info = newdf
                    else:
                        rc.products_info = _pd.concat([rc.products_info, newdf], ignore_index=True).drop_duplicates(subset=['product_id']).reset_index(drop=True)
                    _joblib.dump(rc.products_info, Path(__file__).resolve().parent / 'products.pkl')
                    st.success(f'Fetched and merged {len(new_rows)} items from Amazon')
                    # prefetch their images
                    prefetch_urls([r.get('image_url') for r in new_rows if r.get('image_url')])
            except Exception as e:
                st.error(f'PA-API fetch failed: {e}')
    # Image display toggle
    st.markdown('---')
    show_images = st.checkbox('Show product images', value=st.session_state.get('show_images', True))
    st.session_state.show_images = show_images

# Debug / data inspection panel to help diagnose missing images or labels
with st.sidebar.expander("🧰 Data Debug (columns & sample)"):
    st.write("This panel helps debug why images or names may not appear. It is read-only and safe.")
    if rc.products_info is None:
        st.write("No `products_info` loaded.")
    else:
        try:
            st.write("**Columns:**")
            st.write(list(rc.products_info.columns))
            st.write("**Sample (first 5 rows):**")
            st.dataframe(rc.products_info.head(5), use_container_width=True)

            def detect_image_column(df):
                candidates = ['image_url', 'image', 'img', 'image_link', 'thumbnail', 'imageURL', 'product_image']
                for c in candidates:
                    if c in df.columns:
                        return c
                # fallback: any column containing 'image' or 'img'
                for c in df.columns:
                    if 'image' in c.lower() or 'img' in c.lower():
                        return c
                return None

            img_col = detect_image_column(rc.products_info)
            st.write(f"Detected image column: {img_col or 'None'}")
            if img_col:
                try:
                    sample_imgs = rc.products_info[img_col].head(8).tolist()
                    st.write("Sample image values:")
                    for i, v in enumerate(sample_imgs):
                        st.write(f"{i+1}. {v}")
                except Exception:
                    st.write("Could not read sample image values.")
        except Exception as e:
            st.write(f"Error inspecting products_info: {e}")

# Enhanced the recommender core with additional functionality
# Initialize enhanced product data
enhanced_products = enhance_product_data()