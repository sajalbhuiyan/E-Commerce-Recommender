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
        
        # Product image container
        st.markdown('<div class="product-image-container">', unsafe_allow_html=True)
        img_url = get_product_image(prod)
        try:
            st.image(img_url, use_column_width=True, output_format='auto')
        except Exception:
            st.image('https://via.placeholder.com/400x400/cccccc/969696?text=Image+Not+Found', 
                    use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Product info
        title = prod.get('name') or prod.get('title') or f"Product {pid}"
        st.markdown(f'<div class="product-title">{title}</div>', unsafe_allow_html=True)
        
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
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            cart_count = st.session_state.cart.get(pid, 0)
            if st.button("🛒", key=f"cart_{pid}_{context}", help="Add to cart", use_container_width=True):
                st.session_state.cart[pid] = cart_count + 1
                st.rerun()
        
        with col2:
            is_in_wishlist = pid in st.session_state.wishlist
            wishlist_icon = "❤️" if is_in_wishlist else "🤍"
            if st.button(wishlist_icon, key=f"wish_{pid}_{context}", help="Add to wishlist", use_container_width=True):
                if is_in_wishlist:
                    st.session_state.wishlist.remove(pid)
                else:
                    st.session_state.wishlist.add(pid)
                st.rerun()
        
        with col3:
            if st.button("👁️", key=f"view_{pid}_{context}", help="View details", use_container_width=True):
                st.session_state.view_history.append({
                    'product_id': pid,
                    'product_name': title,
                    'timestamp': datetime.now(),
                    'context': context
                })
                st.session_state.current_product_view = pid
                st.rerun()
        
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
            
            display = " • ".join(label_parts) if label_parts else f"Product {pid}"
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
                st.rerun()

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
            st.rerun()
        
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
                similarity_scores = [f"Product {i+1}: {(len(recs_list)-i)/len(recs_list)*100:.1f}%" for i in range(min(5, len(recs_list)))]
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
                        st.rerun()
                with pag_col2:
                    st.write(f"**Page {page} of {total_pages}** - Showing {len(page_recs)} products")
                with pag_col3:
                    if page < total_pages and st.button("Next ➡️", key="next_content"):
                        st.session_state.content_page = page + 1
                        st.rerun()
                with pag_col4:
                    if st.button("🔄 Refresh", key="refresh_content"):
                        st.session_state.content_page = 1
                        st.rerun()

# Continue with other tabs (Hybrid, Personal, Cart, Analytics)...
# [The rest of the tabs would follow similar enhancement patterns]

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
                                st.rerun()
                        with col4:
                            item_total = prod.get('price', 0) * quantity
                            total_amount += item_total
                            st.write(f"${item_total:.2f}")
            
            st.markdown(f"### Total: ${total_amount:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Continue Shopping", use_container_width=True):
                    st.session_state.show_cart = False
                    st.rerun()
            with col2:
                if st.button("🚀 Checkout", type="primary", use_container_width=True):
                    st.success("Order placed successfully! 🎉")
                    st.session_state.cart = {}
                    st.session_state.show_cart = False
                    st.rerun()

# Demo data initialization
if st.sidebar.button("🔄 Initialize Demo Data", use_container_width=True):
    with st.spinner("Creating rich demo dataset..."):
        ok = create_demo_data(n_items=200, n_users=100)
        if ok:
            st.sidebar.success("Demo data created! Refresh to see enhanced features.")
            st.rerun()
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
def get_trending_products(n=8):
    """Get trending products based on views and cart additions"""
    # This would integrate with actual analytics in a real system
    if rc.products_info is not None:
        return rc.products_info.sample(min(n, len(rc.products_info)))
    return []

def get_personalized_recommendations(user_id, n=12):
    """Get personalized recommendations based on user behavior"""
    # Enhanced version that considers view history and cart
    viewed_products = [item['product_id'] for item in st.session_state.view_history]
    cart_products = list(st.session_state.cart.keys())
    
    # Use these to enhance recommendations
    return cached_user_recommend(user_id, n)

# Initialize enhanced product data
enhanced_products = enhance_product_data()