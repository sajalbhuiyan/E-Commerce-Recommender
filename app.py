import streamlit as st
from recommender_core import (
    content_recommend,
    hybrid_recommend,
    user_recommend,
    cached_content_recommend,
    cached_hybrid_recommend,
    cached_user_recommend,
    get_product_choices,
    fuzzy_find,
    explain_tfidf,
    create_demo_data,
    products_info,
    item_emb,
    fetch_artifacts_from_env,
)
import pandas as pd
import math


st.set_page_config(page_title="E-Commerce Recommender", layout="wide")

# Try to fetch configured artifacts (S3 URLs or env) now that page config
# has been set and Streamlit has been initialized. This avoids calling
# Streamlit functions during module import which would break set_page_config.
try:
    fetch_artifacts_from_env()
except Exception:
    # keep app resilient — artifact fetch may fail in offline dev
    pass

# Sidebar / settings
st.sidebar.title("E-Commerce Recommender")
algorithm = st.sidebar.selectbox("Select Recommendation Algorithm:", ["Content-Based", "Hybrid"])
top_n = st.sidebar.slider("Number of Recommendations:", 1, 20, 10)


# Main UI
tab1, tab2 = st.tabs(["Product Recommendations", "Hybrid Recommendations"])

with tab1:
    st.header("Content-Based Recommendations")
    choices = get_product_choices(200)
    if choices:
        product_choice = st.selectbox("Choose a product (or type an id/name):", choices)
        product_input = st.text_input("Or type product id / name:")
        # show fuzzy suggestions below the input when typing
        if product_input:
            matches = fuzzy_find(product_input, n=5)
            if matches:
                st.write("Did you mean:", ", ".join(matches))
        query = product_input.strip() or product_choice
    else:
        query = st.text_input("Enter Product ID or Name:")

    if st.button("Get Recommendations", key="content"):
        if not query:
            st.warning("Please enter a product id, name, or brand to search.")
        else:
            recs = content_recommend(query, top_n)
            if not recs:
                st.info("No recommendations found. Try a different product or add embeddings/metadata to the project.")
            else:
                st.write(f"Top {top_n} similar products for '{query}':")
                # use polished card + pagination display
                def paginate(items, page, per_page):
                    total = len(items)
                    pages = max(1, math.ceil(total / per_page))
                    start = (page - 1) * per_page
                    end = start + per_page
                    return items[start:end], pages

                def show_product_card(pid):
                    # Render a compact product card with image, title, price and an expander for details
                    if products_info is not None and 'product_id' in products_info.columns:
                        pr = products_info[products_info['product_id'] == pid]
                        if not pr.empty:
                            prod = pr.iloc[0]
                            # Robust image lookup: support several common column names and list-like values
                            img = None
                            for col in ('image_url', 'image', 'img', 'image_link', 'thumbnail'):
                                if col in prod.index:
                                    img = prod.get(col)
                                    break
                            # If image is an array or list, pick the first element
                            try:
                                import numpy as _np
                                if isinstance(img, (list, tuple, _np.ndarray)):
                                    img = img[0] if len(img) > 0 else None
                            except Exception:
                                if isinstance(img, (list, tuple)):
                                    img = img[0] if len(img) > 0 else None

                            # Fallback placeholder for missing images
                            if img is None or (isinstance(img, str) and img.strip() == '') or pd.isna(img):
                                img = 'https://via.placeholder.com/160?text=No+Image'
                            try:
                                st.image(img, width=160)
                            except Exception:
                                # If image URL fails for any reason, show placeholder
                                st.image('https://via.placeholder.com/160?text=No+Image', width=160)

                            title = prod.get('name') or prod.get('product_id') or pid
                            st.markdown(f"**{title}**")
                            if 'brand' in prod.index:
                                st.caption(f"{prod['brand']}")
                            if 'price' in prod.index:
                                st.write(f"**${prod['price']}**")
                            with st.expander("Details"):
                                st.write(prod.to_dict())
                                # TF-IDF explanation when available
                                feats = explain_tfidf(pid, top_k=8)
                                if feats:
                                    st.write("Top TF-IDF features:")
                                    st.write(", ".join(feats))
                            return
                    # fallback
                    st.write(f"**ID:** {pid}")

                per_page = 8
                page = st.session_state.get('content_page', 1)
                # use cached wrapper for better performance
                recs_cached = list(cached_content_recommend(str(query), top_n))
                items, total_pages = paginate(recs_cached, page, per_page)
                # render grid
                cols = st.columns(4)
                for i, pid in enumerate(items):
                    with cols[i % 4]:
                        show_product_card(pid)

                # download current page
                import io, csv
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(['product_id'])
                for pid in items:
                    writer.writerow([pid])
                st.download_button(label='Download recommendations (page)', data=csv_buf.getvalue(), file_name='recs_page.csv', mime='text/csv')

                # pagination controls
                col1, col2, col3 = st.columns([1, 6, 1])
                with col1:
                    if st.button("Prev", key='prev_content'):
                        st.session_state['content_page'] = max(1, page - 1)
                        st.experimental_rerun()
                with col3:
                    if st.button("Next", key='next_content'):
                        st.session_state['content_page'] = min(total_pages, page + 1)
                        st.experimental_rerun()
                with col2:
                    st.write(f"Page {page} / {total_pages}")

            # show TF-IDF explanation when available (global)
            if st.button("Explain (TF-IDF) this product", key="explain"):
                features = explain_tfidf(query, top_k=10)
                if features:
                    st.write("Top TF-IDF features for this product:")
                    st.write(", ".join(features))
                else:
                    st.info("TF-IDF artifacts not available to explain this product.")

with tab2:
    st.header("Hybrid Recommendations")
    user_id = st.text_input("Enter User ID (optional):")
    if st.button("Get Hybrid Recommendations", key="hybrid"):
        recs = hybrid_recommend(user_id or None, top_n)
        if not recs:
            st.info("No hybrid recommendations available with current artifacts. Showing defaults if possible.")
        else:
            st.write(f"Top {top_n} hybrid recommendations:")
            # simple paginated card display
            per_page = 8
            page = st.session_state.get('hybrid_page', 1)
            def paginate(items, page, per_page):
                total = len(items)
                pages = max(1, math.ceil(total / per_page))
                start = (page - 1) * per_page
                end = start + per_page
                return items[start:end], pages

            def show_card(pid):
                if products_info is not None and 'product_id' in products_info.columns:
                    pr = products_info[products_info['product_id'] == pid]
                    if not pr.empty:
                        prod = pr.iloc[0]
                        # Robust image handling as above
                        img = None
                        for col in ('image_url', 'image', 'img', 'image_link', 'thumbnail'):
                            if col in prod.index:
                                img = prod.get(col)
                                break
                        try:
                            import numpy as _np
                            if isinstance(img, (list, tuple, _np.ndarray)):
                                img = img[0] if len(img) > 0 else None
                        except Exception:
                            if isinstance(img, (list, tuple)):
                                img = img[0] if len(img) > 0 else None
                        if img is None or (isinstance(img, str) and img.strip() == '') or pd.isna(img):
                            img = 'https://via.placeholder.com/160?text=No+Image'
                        try:
                            st.image(img, width=160)
                        except Exception:
                            st.image('https://via.placeholder.com/160?text=No+Image', width=160)

                        st.markdown(f"**{prod.get('name') or pid}**")
                        if 'price' in prod.index:
                            st.write(f"**${prod['price']}**")
                        with st.expander('Details'):
                            st.write(prod.to_dict())
                        return
                st.write(f"**ID:** {pid}")

            # use cached hybrid call
            recs_cached = list(cached_hybrid_recommend(user_id or None, top_n))
            items, total_pages = paginate(recs_cached, page, per_page)
            cols = st.columns(4)
            for i, pid in enumerate(items):
                with cols[i % 4]:
                    show_card(pid)
            # download full hybrid recommendations
            import io, csv
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(['product_id'])
            for pid in recs:
                writer.writerow([pid])
            st.download_button(label='Download full hybrid recommendations', data=csv_buf.getvalue(), file_name='hybrid_recs.csv', mime='text/csv')

            c1, c2, c3 = st.columns([1,6,1])
            with c1:
                if st.button('Prev', key='prev_hybrid'):
                    st.session_state['hybrid_page'] = max(1, page - 1)
                    st.experimental_rerun()
            with c3:
                if st.button('Next', key='next_hybrid'):
                    st.session_state['hybrid_page'] = min(total_pages, page + 1)
                    st.experimental_rerun()
            with c2:
                st.write(f"Page {page} / {total_pages}")

    # quick user-recommendation helper
    st.write("---")
    st.markdown("**Get recommendations for a specific user (if user-item data exists)**")
    user_query = st.text_input("User ID for recommendations:")
    if st.button("Recommend for user", key="userrec"):
        if not user_query:
            st.warning("Please enter a user id.")
        else:
            recs = user_recommend(user_query, top_n)
            if not recs:
                st.info("No user recommendations available for this id.")
            else:
                # reuse hybrid card layout
                per_page = 8
                page = st.session_state.get('user_page', 1)
                recs_cached = list(cached_user_recommend(user_query, top_n))
                items, total_pages = paginate(recs_cached, page, per_page)
                cols = st.columns(4)
                for i, pid in enumerate(items):
                    with cols[i % 4]:
                        show_card(pid)
                c1, c2, c3 = st.columns([1,6,1])
                with c1:
                    if st.button('Prev', key='prev_user'):
                        st.session_state['user_page'] = max(1, page - 1)
                        st.experimental_rerun()
                with c3:
                    if st.button('Next', key='next_user'):
                        st.session_state['user_page'] = min(total_pages, page + 1)
                        st.experimental_rerun()
                with c2:
                    st.write(f"Page {page} / {total_pages}")

st.sidebar.markdown("---")
st.sidebar.write("Status:")
st.sidebar.write(f"- products.pkl: {'loaded' if products_info is not None else 'missing'}")
st.sidebar.write(f"- item_emb.pkl: {'loaded' if item_emb is not None else 'missing'}")
st.sidebar.success("App ready — the UI will degrade gracefully if some artifacts are missing.")
st.sidebar.markdown("---")
if st.sidebar.button("Create demo data (small)"):
    ok = create_demo_data(n_items=80, n_users=30)
    if ok:
        st.sidebar.success("Demo data created — refresh the app to use it.")
    else:
        st.sidebar.error("Failed to create demo data.")
# Redis connection controls
st.sidebar.markdown('---')
st.sidebar.markdown('### Redis (optional)')
redis_host = st.sidebar.text_input('Redis host', value='localhost')
redis_port = st.sidebar.number_input('Redis port', value=6379)
redis_db = st.sidebar.number_input('Redis DB', value=0)
redis_pass = st.sidebar.text_input('Redis password (optional)', value='', type='password')
if st.sidebar.button('Connect Redis'):
    try:
        from recommender_core import init_redis
        ok = init_redis(host=redis_host, port=int(redis_port), db=int(redis_db), password=redis_pass or None)
        if ok:
            st.sidebar.success('Connected to Redis')
        else:
            st.sidebar.error('Failed to connect to Redis')
    except Exception as e:
        st.sidebar.error(f'Error: {e}')

# Remote artifact upload (S3/HTTP)
st.sidebar.markdown('---')
st.sidebar.markdown('### Download artifact from URL')
remote_url = st.sidebar.text_input('Artifact URL (S3 or HTTP)')
remote_filename = st.sidebar.text_input('Save as filename (e.g. products.pkl)')
if st.sidebar.button('Download artifact'):
    if not remote_url or not remote_filename:
        st.sidebar.error('Provide URL and filename')
    else:
        try:
            from recommender_core import fetch_remote_artifact
            ok = fetch_remote_artifact(remote_url, remote_filename)
            if ok:
                st.sidebar.success('Downloaded and registered artifact')
            else:
                st.sidebar.error('Failed to download or checksum mismatch')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')

    # Model tools: validate and convert to ONNX
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Model tools')
    model_pkl = st.sidebar.text_input('Model filename (e.g. rf_recommender.pkl)')
    if st.sidebar.button('Validate model'):
        if not model_pkl:
            st.sidebar.error('Enter model filename')
        else:
            try:
                from recommender_core import validate_and_load_model
                ok, out = validate_and_load_model(model_pkl)
                if ok:
                    st.sidebar.success(f'Loaded model: {type(out)}')
                else:
                    st.sidebar.error(f'Load failed: {out}')
            except Exception as e:
                st.sidebar.error(f'Error: {e}')

    if st.sidebar.button('Convert to ONNX'):
        if not model_pkl:
            st.sidebar.error('Enter model filename')
        else:
            try:
                from recommender_core import convert_sklearn_to_onnx
                ok, out = convert_sklearn_to_onnx(model_pkl)
                if ok:
                    st.sidebar.success(f'ONNX written: {out}')
                else:
                    st.sidebar.error(f'Conversion failed: {out}')
            except Exception as e:
                st.sidebar.error(f'Error: {e}')
