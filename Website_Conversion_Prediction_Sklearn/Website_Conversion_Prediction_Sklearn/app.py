import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Website Conversion Prediction",
    page_icon="🎯",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #047857;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-convert {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .badge-drop {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    .metric-banner {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Website Conversion & CRO Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict e-commerce visitor purchase conversions and trigger dynamic retention incentives using <b>RandomForestClassifier</b>.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models" / "website_conversion_model.pkl"
csv_path = BASE_DIR / "data" / "website_conversion.csv"
chart_path = BASE_DIR / "outputs" / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Live Visitor Session Evaluator", "📈 Feature Importance & Drivers", "📋 Historical Visitor Sessions"])

source_map = {1: "Direct Traffic", 2: "Organic Search (Google/Bing)", 3: "Social Media / Referral"}
device_map = {1: "Desktop Computer", 2: "Mobile Smartphone", 3: "Tablet"}

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Visitor Session Attributes")
        
        persona = st.selectbox(
            "⚡ Quick Visitor Persona Preset",
            ["Custom Session", "🛒 High-Intent Returning Buyer", "👀 Casual Mobile Browser (High Bounce Risk)", "🔍 Search Engine Organic Researcher", "⚡ Fast Direct Checkout"]
        )
        
        if persona == "🛒 High-Intent Returning Buyer":
            def_dur, def_pages, def_prev, def_ret, def_cart, def_src, def_dev = 480, 9, 4, 1, 3, 1, 1
        elif persona == "👀 Casual Mobile Browser (High Bounce Risk)":
            def_dur, def_pages, def_prev, def_ret, def_cart, def_src, def_dev = 65, 2, 0, 0, 0, 3, 2
        elif persona == "🔍 Search Engine Organic Researcher":
            def_dur, def_pages, def_prev, def_ret, def_cart, def_src, def_dev = 300, 6, 2, 1, 1, 2, 1
        elif persona == "⚡ Fast Direct Checkout":
            def_dur, def_pages, def_prev, def_ret, def_cart, def_src, def_dev = 360, 8, 3, 1, 2, 1, 1
        else:
            def_dur, def_pages, def_prev, def_ret, def_cart, def_src, def_dev = 420, 8, 3, 1, 2, 2, 1
            
        with st.form("cro_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                duration = st.slider("Session Duration (Seconds)", 10, 1200, int(def_dur), step=10, format="%d s")
                pages_viewed = st.slider("Pages Viewed in Session", 1, 25, int(def_pages))
                previous_visits = st.slider("Prior Site Visits", 0, 10, int(def_prev))
                is_returning = st.radio("Visitor Type", [1, 0], index=0 if def_ret == 1 else 1, format_func=lambda x: "Returning Visitor" if x == 1 else "First-Time Visitor")
            with col_b:
                cart_items = st.slider("Items Added to Shopping Cart", 0, 10, int(def_cart))
                traffic_source = st.selectbox("Traffic Acquisition Channel", options=[1, 2, 3], index=[1, 2, 3].index(def_src), format_func=lambda x: source_map[x])
                device_type = st.selectbox("Browsing Device", options=[1, 2, 3], index=[1, 2, 3].index(def_dev), format_func=lambda x: device_map[x])
                
            submit_btn = st.form_submit_button("🚀 Evaluate Conversion Probability", use_container_width=True)
            
    with col_result:
        st.subheader("Conversion Prediction & CRO Guidance")
        if submit_btn:
            visitor = pd.DataFrame([{
                "session_duration_seconds": duration,
                "pages_viewed": pages_viewed,
                "previous_visits": previous_visits,
                "is_returning_visitor": is_returning,
                "cart_items": cart_items,
                "traffic_source": traffic_source,
                "device_type": device_type
            }])
            
            pred = model.predict(visitor)[0]
            prob = model.predict_proba(visitor)[0][1]
            
            if pred == 1:
                st.markdown(f"""
                <div class="badge-convert">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Conversion Forecast</div>
                    <div class="metric-banner">🎉 HIGH LIKELIHOOD TO CONVERT</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Conversion Probability: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="badge-drop">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Conversion Forecast</div>
                    <div class="metric-banner">⚠️ CART ABANDONMENT RISK</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Conversion Likelihood: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📊 Intent Score")
            st.progress(float(prob))
            
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Conversion Probability", f"{prob:.1%}")
            col_c2.metric("Drop-off Likelihood", f"{(1 - prob):.1%}")
            
            st.markdown("### 🛠️ Automated CRO Nudges")
            nudges = []
            if cart_items > 0 and prob < 0.6:
                nudges.append("🏷️ **Exit-Intent Recovery**: Cart items detected with high abandonment risk. Trigger a timed 10% discount overlay before tab close.")
            if duration >= 300 and pages_viewed >= 6:
                nudges.append("💬 **Live Help Intervention**: Visitor exhibits deep consideration. Open automated chatbot to answer product questions.")
            if is_returning == 0:
                nudges.append("🎁 **New Visitor Welcome**: Offer free shipping on first order to lower barrier to purchase.")
            if pred == 1:
                nudges.append("⚡ **Streamlined Checkout**: Suppress non-essential upsell popups to ensure friction-free transaction.")
                
            for n in nudges:
                st.write(n)
                
            with st.expander("🔍 Model Input Payload"):
                st.json(visitor.to_dict(orient="records")[0])
        else:
            st.info("👈 Configure visitor behavior metrics and click **'Evaluate Conversion Probability'**.")

with tab2:
    st.subheader("Feature Importance Breakdown")
    if chart_path.exists():
        st.image(str(chart_path), caption="Top Signals Driving Website Purchases", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Historical Sessions Dataset (website_conversion.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", f"{len(df):,}")
        col2.metric("Baseline Conversion Rate", f"{df['converted'].mean():.1%}")
        col3.metric("Avg Session Duration", f"{df['session_duration_seconds'].mean():.0f} sec")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Dataset not found.")
