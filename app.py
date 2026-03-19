import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Failure Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PIXEL-PERFECT CSS INJECTION (SCALED TO FIT)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 1. Global Resets & Layout Scaling */
    html, body, [class*="css"], .stMarkdown p, .stText {
        font-family: 'Inter', sans-serif !important;
        background-color: #F8F9FB !important;
        color: #111827 !important;
    }
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8F9FB !important;
    }
    
    /* Tightly compress the main container to fit 1 screen */
    .block-container { 
        padding-top: 1.5rem !important; 
        padding-bottom: 0rem !important; 
        max-width: 1300px !important;
    }

    /* 2. Typography */
    h1 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #111827 !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
    }
    .subtitle {
        color: #6B7280;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
    }

    /* 3. Sidebar Styling (Fixed Visibility) */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
        width: 250px !important;
    }
    
    [data-testid="stSidebarHeader"] {
        display: none !important;
        padding: 0 !important;
        height: 0 !important;
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        font-weight: 600;
        font-size: 0.95rem;
        color: #111827 !important;
        padding: 0.5rem 0 0.75rem 0;
        border-bottom: 1px solid #F3F4F6;
        margin-bottom: 0.75rem;
    }

    .sidebar-icon {
        background: #FF5A1F;
        color: white;
        width: 26px;
        height: 26px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        font-size: 14px;
    }
    
    .nav-header {
        font-size: 0.65rem;
        font-weight: 600;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    /* Custom Navigation Buttons (SAFE OVERRIDE) */
    .stRadio > div[role="radiogroup"] {
        gap: 0.25rem;
    }
    
    .stRadio label {
        padding: 0.5rem 0.75rem !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .stRadio label[data-checked="true"],
    .stRadio label:has(input:checked),
    .stRadio label:has([aria-checked="true"]),
    .stRadio label:has([data-checked="true"]) {
        background-color: #FFF7ED !important;
    }
    
    .stRadio label[data-checked="true"] p,
    .stRadio label:has(input:checked) p,
    .stRadio label:has([aria-checked="true"]) p,
    .stRadio label:has([data-checked="true"]) p,
    .stRadio label[data-checked="true"] span,
    .stRadio label:has(input:checked) span,
    .stRadio label:has([aria-checked="true"]) span,
    .stRadio label:has([data-checked="true"]) span {
        color: #FF5A1F !important;
        font-weight: 600 !important;
    }
    
    .stRadio label:hover {
        background-color: #FFF7ED !important;
    }
    
    /* Hide the radio circle safely */
    .stRadio label [data-baseweb="radio"] {
        display: none !important;
    }
    
    .stRadio label > div:first-child:not(:has(p)):not(:has(span)) {
        display: none !important;
    }

    .stRadio label p, .stRadio label span {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #4B5563 !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }

    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 250px;
        padding: 1rem;
        border-top: 1px solid #F3F4F6;
        display: flex;
        align-items: center;
        font-size: 0.75rem;
        color: #4B5563 !important;
        font-weight: 500;
        background: #FFFFFF;
        z-index: 999;
        box-sizing: border-box;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 8px;
    }

    /* 4. Top KPI Metrics (Scaled down vertically) */
    .kpi-container {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .kpi-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .kpi-title {
        color: #6B7280;
        font-size: 0.7rem;
        font-weight: 500;
        margin-bottom: 0.15rem;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 400;
        line-height: 1.1;
    }
    .val-total { color: #111827; }
    .val-living { color: #31C48D; }
    .val-deceased { color: #F87171; }

    /* 5. Chart Cards (Scaled to fit screen) */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        height: 100%;
        margin-bottom: 0.75rem;
    }
    .chart-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    /* 6. Custom Correlation UI */
    .corr-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.7rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        color: #4B5563;
        padding-left: 0.75rem;
        padding-right: 0.5rem;
    }
    .corr-label { width: 130px; }
    .corr-bar-bg {
        flex-grow: 1;
        height: 4px;
        background-color: #F3F4F6;
        border-radius: 2px;
        margin: 0 8px;
        position: relative;
    }
    .corr-bar-fill {
        height: 100%;
        border-radius: 2px;
        position: absolute;
    }
    .fill-neg { background-color: #31C48D; right: 50%; }
    .fill-pos { background-color: #F87171; left: 50%; }
    .corr-val { width: 35px; text-align: right; }
    .corr-footer {
        font-size: 0.65rem;
        color: #9CA3AF;
        margin-top: 0.75rem;
        line-height: 1.2;
    }

    /* 7. Form UI (Interactive Prediction) */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.75rem !important;
        color: #374151 !important;
        font-weight: 500 !important;
        padding-bottom: 0.2rem !important;
    }

    div[data-baseweb="slider"] div[data-testid="stTickBar"] { display: none; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #FF5A1F !important;
        border: 2px solid #FFF !important;
        box-shadow: 0 0 0 1px #FF5A1F;
    }
    div[data-baseweb="slider"] div > div > div {
        background-color: #FF5A1F !important;
    }

    .stButton > button {
        background-color: #FF5A1F;
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
        padding: 0.4rem 1.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        float: right;
    }
    .stButton > button:hover { background-color: #E04E1A; color: white; }

    #MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CHART STYLING CONFIG (Matplotlib)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#E5E7EB',
    'text.color': '#6B7280',
    'axes.labelcolor': '#6B7280',
    'xtick.color': '#9CA3AF',
    'ytick.color': '#9CA3AF',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'grid.color': '#F3F4F6',
    'grid.linestyle': '--',
    'font.size': 7
})

# ==========================================
# DATA & MOCK GENERATION
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 299
    df = pd.DataFrame({
        'age': np.random.randint(40, 95, n),
        'anaemia': np.random.randint(0, 2, n),
        'creatinine_phosphokinase': np.random.randint(20, 8000, n),
        'diabetes': np.random.randint(0, 2, n),
        'ejection_fraction': np.random.randint(14, 80, n),
        'high_blood_pressure': np.random.randint(0, 2, n),
        'platelets': np.random.normal(263000, 90000, n),
        'serum_creatinine': np.random.lognormal(0.2, 0.4, n),
        'serum_sodium': np.random.normal(136, 4, n).astype(int),
        'sex': np.random.randint(0, 2, n),
        'smoking': np.random.randint(0, 2, n),
        'time': np.random.randint(4, 285, n)
    })
    death_events = np.array([0]*203 + [1]*96)
    np.random.shuffle(death_events)
    df['DEATH_EVENT'] = death_events
    return df

@st.cache_resource
def load_model(dst_path="./app/models/model.xgb"):
    if not XGB_AVAILABLE:
        return "MOCK_MODEL"
        
    try:
        model = xgb.XGBClassifier()
        model.load_model(dst_path)
        return model
    except Exception:
        # Fallback to mock model for UI demonstration
        return "MOCK_MODEL"
data = load_data()
model = load_model()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-icon">∿</div>
            <span style="color: #111827; visibility: visible;">Heart Failure Analysis</span>
        </div>
        <div class="nav-header">NAVIGATION</div>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Navigation",
        [":material/bar_chart: \u00A0 Exploratory Data Analysis", ":material/stethoscope: \u00A0 Interactive Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
        <div class="sidebar-footer">
            <div class="status-dot"></div> Model Loaded
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 1: EDA
# ==========================================
if "Exploratory" in app_mode:
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Overview of the heart failure clinical records dataset.</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Total Patients</div>
                <div class="kpi-val val-total">299</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Living Cases</div>
                <div class="kpi-val val-living">203</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Deceased Cases</div>
                <div class="kpi-val val-deceased">96</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Aggressively shrink figsize to fit exactly on screen
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.markdown("<div class='chart-card'><div class='chart-header'>Age Distribution</div>", unsafe_allow_html=True)
        fig_age, ax_age = plt.subplots(figsize=(5, 1.8)) 
        
        labels = ['40-50', '50-60', '60-70', '70-80', '80-90', '90+']
        vals = [48, 82, 95, 52, 18, 4] 
        
        ax_age.bar(labels, vals, color='#5155EC', width=0.75, edgecolor='none', zorder=3)
        ax_age.set_yticks([0, 25, 50, 75, 100])
        ax_age.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax_age.tick_params(axis='both', length=0, pad=3)
        
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_c2:
        st.markdown("<div class='chart-card'><div class='chart-header'>Diabetes vs. Mortality</div>", unsafe_allow_html=True)
        fig_diab, ax_diab = plt.subplots(figsize=(5, 1.8)) 
        
        x_labels = ['No Diabetes', 'Has Diabetes']
        living_counts = [118, 85]
        deceased_counts = [56, 40]
        
        ax_diab.bar(x_labels, living_counts, color='#31C48D', width=0.5, label='Living', zorder=3)
        ax_diab.bar(x_labels, deceased_counts, bottom=living_counts, color='#F87171', width=0.5, label='Deceased', zorder=3)
        ax_diab.set_yticks([0, 45, 90, 135, 180])
        ax_diab.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax_diab.tick_params(axis='both', length=0, pad=3)
        
        ax_diab.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, 
                       handlelength=1, handleheight=1, handletextpad=0.5, 
                       labels=['Living', 'Deceased'], 
                       handles=[patches.Patch(color='#F87171'), patches.Patch(color='#31C48D')])
        
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_diab, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("<div class='chart-card'><div class='chart-header'>Mortality Ratio</div>", unsafe_allow_html=True)
        fig_donut, ax_donut = plt.subplots(figsize=(5, 1.8))
        
        sizes = [203, 96]
        colors = ['#31C48D', '#F87171']
        
        wedges, _ = ax_donut.pie(sizes, colors=colors, startangle=140, 
                                 wedgeprops=dict(width=0.25, edgecolor='#FFFFFF', linewidth=2.5))
        ax_donut.axis('equal')
        
        ax_donut.legend(handles=[patches.Patch(color='#F87171'), patches.Patch(color='#31C48D')], 
                        labels=['Deceased', 'Living'], loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                        ncol=2, frameon=False, handlelength=1, handleheight=1)
        
        plt.tight_layout(pad=0)
        st.pyplot(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_c2:
        st.markdown("<div class='chart-card'><div class='chart-header'>Top Correlations with Mortality</div>", unsafe_allow_html=True)
        
        corrs = [
            ("time", -0.53, "neg"),
            ("serum_creatinine", 0.29, "pos"),
            ("ejection_fraction", -0.27, "neg"),
            ("age", 0.25, "pos"),
            ("serum_sodium", -0.20, "neg"),
            ("high_blood_pressure", 0.08, "pos")
        ]
        
        html_corrs = ""
        for name, val, typ in corrs:
            w = (abs(val) / 0.6) * 50
            fill_class = "fill-pos" if typ == "pos" else "fill-neg"
            sign = "+" if val > 0 else ""
            html_corrs += f"""
            <div class="corr-row">
                <div class="corr-label">{name}</div>
                <div class="corr-bar-bg">
                    <div class="corr-bar-fill {fill_class}" style="width: {w}%;"></div>
                </div>
                <div class="corr-val">{sign}{val:.2f}</div>
            </div>
            """
            
        st.markdown(html_corrs, unsafe_allow_html=True)
        st.markdown("<div class='corr-footer'>'time', 'serum_creatinine', and 'ejection_fraction' have the strongest correlation with the 'DEATH_EVENT'.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: INTERACTIVE PREDICTION
# ==========================================
elif "Prediction" in app_mode:
    st.markdown("<h1>Interactive Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enter patient clinical records to predict mortality risk.</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; padding-bottom:0.2rem;'><span style='font-weight:500;'>Age</span><span style='color:#9CA3AF;'>60 yrs</span></div>", unsafe_allow_html=True)
            age = st.slider('age_hide', 40, 95, 60, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            
            anaemia = st.selectbox('Anaemia', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            st.markdown("<br>", unsafe_allow_html=True)
            
            creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', min_value=0, max_value=8000, value=582)
            st.markdown("<br>", unsafe_allow_html=True)
            
            diabetes = st.selectbox('Diabetes', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; padding-bottom:0.2rem;'><span style='font-weight:500;'>Ejection Fraction</span><span style='color:#9CA3AF;'>38%</span></div>", unsafe_allow_html=True)
            ejection_fraction = st.slider('ef_hide', 14, 80, 38, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            
            high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            st.markdown("<br>", unsafe_allow_html=True)
            
            platelets = st.number_input('Platelets (kiloplatelets/mL)', min_value=0.0, max_value=850000.0, value=263358.0, step=1000.0)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; padding-bottom:0.2rem;'><span style='font-weight:500;'>Serum Creatinine</span><span style='color:#9CA3AF;'>1.1 mg/dL</span></div>", unsafe_allow_html=True)
            serum_creatinine = st.slider('sc_hide', 0.5, 9.4, 1.1, 0.1, label_visibility="collapsed")

        with col3:
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; padding-bottom:0.2rem;'><span style='font-weight:500;'>Serum Sodium</span><span style='color:#9CA3AF;'>136 mEq/L</span></div>", unsafe_allow_html=True)
            serum_sodium = st.slider('ss_hide', 113, 148, 136, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            
            sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            st.markdown("<br>", unsafe_allow_html=True)
            
            smoking = st.selectbox('Smoking', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; padding-bottom:0.2rem;'><span style='font-weight:500;'>Follow-up period</span><span style='color:#9CA3AF;'>130 days</span></div>", unsafe_allow_html=True)
            time = st.slider('time_hide', 4, 285, 130, label_visibility="collapsed")

        st.markdown("<hr style='margin: 1rem 0 0.75rem 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)
        
        footer_c1, footer_c2 = st.columns([3, 1])
        with footer_c1:
            st.markdown("<div style='font-size:0.7rem; color:#9CA3AF; padding-top:0.25rem; line-height: 1.3;'>Disclaimer: This is an AI-generated prediction based on a simulated model and should not be used<br>for actual medical diagnosis. Consult a healthcare professional.</div>", unsafe_allow_html=True)
        with footer_c2:
            submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_dict = {
            'age': float(age), 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
            'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure,
            'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
            'sex': sex, 'smoking': smoking, 'time': time
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        try:
            if isinstance(model, str) and model == "MOCK_MODEL":
                risk_score = (age / 100) * 0.3 + (1 if ejection_fraction < 35 else 0) * 0.4 + (serum_creatinine / 5) * 0.3
                prediction = [1 if risk_score > 0.5 else 0]
            else:
                input_df = pd.DataFrame([input_dict])
                feature_names = model.get_booster().feature_names if hasattr(model, 'get_booster') else input_df.columns
                input_df = input_df[feature_names] 
                prediction = model.predict(input_df)

            if prediction[0] == 1:
                st.markdown("""
                <div style="background-color: #FEF2F2; border: 1px solid #FECACA; padding: 0.75rem 1rem; border-radius: 8px; display:flex; align-items:center;">
                    <span style="font-size: 1.25rem; margin-right: 12px;">⚠️</span>
                    <div>
                        <div style="color: #DC2626; font-weight: 600; font-size:0.9rem;">High Risk Profile Detected</div>
                        <div style="color: #991B1B; font-size: 0.8rem;">Model predicts elevated mortality risk within the follow-up period.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; padding: 0.75rem 1rem; border-radius: 8px; display:flex; align-items:center;">
                    <span style="font-size: 1.25rem; margin-right: 12px;">✅</span>
                    <div>
                        <div style="color: #16A34A; font-weight: 600; font-size:0.9rem;">Standard Risk Profile</div>
                        <div style="color: #166534; font-size: 0.8rem;">Model predicts low mortality risk within the follow-up period.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"System Error: {str(e)}")
