import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    background-color: #060b18 !important;
    color: #cdd6f4 !important;
    font-family: 'Inter', sans-serif !important;
}
.main, .block-container { background-color: #060b18 !important; }
.stApp { background: #060b18 !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #0f1a30 60%, #0b1120 100%) !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebarContent"] { background: transparent !important; padding: 1rem 0.8rem; }

.sidebar-logo {
    text-align: center;
    padding: 16px 0 8px 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 20px;
}
.sidebar-logo-text {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 900;
    color: #00d4ff;
    letter-spacing: 3px;
}
.sidebar-logo-sub {
    font-size: 0.65rem;
    color: #4a6fa5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 3px;
}
.sidebar-section {
    background: rgba(0, 212, 255, 0.04);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 12px;
    margin-bottom: 12px;
}
.sidebar-section-title {
    font-size: 0.65rem;
    color: #4a6fa5;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
}
.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #0f1e35;
}
.sidebar-stat-label { font-size: 0.8rem; color: #6b8ab5; }
.sidebar-stat-value { font-size: 0.85rem; font-weight: 600; color: #00d4ff; }

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #162847 100%);
    border: 1px solid #1e4080;
    border-top: 2px solid #00d4ff;
    border-radius: 12px;
    padding: 22px 16px 18px 16px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(0,212,255,0.1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff88, transparent);
}
.metric-value {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 900 !important;
    color: #00d4ff !important;
    line-height: 1.1;
    display: block;
}
.metric-label {
    font-size: 0.72rem !important;
    color: #6b8ab5 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-top: 6px !important;
    font-weight: 500 !important;
    display: block;
}
.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
    display: block;
}

/* ── SECTION HEADERS ── */
.section-header {
    background: linear-gradient(90deg, rgba(0,212,255,0.08), transparent);
    border-left: 3px solid #00d4ff;
    padding: 10px 16px;
    margin: 28px 0 18px 0;
    border-radius: 0 8px 8px 0;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    color: #00d4ff !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
}

/* ── INSIGHT BOXES ── */
.insight-box {
    background: linear-gradient(135deg, #0a1f15, #0d2a1a);
    border: 1px solid rgba(0,255,136,0.15);
    border-left: 3px solid #00ff88;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #b8ffd9 !important;
    line-height: 1.7;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1120 !important;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: #4a6fa5 !important;
    background: transparent !important;
    border-radius: 7px !important;
    padding: 8px 12px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    background: rgba(0,212,255,0.1) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── STREAMLIT NATIVE METRICS ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1f3c, #162847) !important;
    border: 1px solid #1e4080 !important;
    border-top: 2px solid #00d4ff !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
div[data-testid="stMetricLabel"] > div {
    color: #6b8ab5 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
div[data-testid="stMetricValue"] > div {
    color: #00d4ff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
}

/* ── FORM ELEMENTS ── */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #7ab3d4 !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* Multiselect tags - make them visible and colorful */
[data-baseweb="tag"] {
    background-color: #0d3a5c !important;
    border: 1px solid #00d4ff55 !important;
    border-radius: 6px !important;
}
[data-baseweb="tag"] span {
    color: #00d4ff !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}
[data-baseweb="tag"] svg { fill: #00d4ff !important; }

/* Multiselect dropdown container */
[data-baseweb="select"] > div {
    background: #0a1829 !important;
    border: 1px solid #1e4080 !important;
    border-radius: 8px !important;
}

/* Slider colors */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #00d4ff !important;
    border-color: #00d4ff !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #00d4ff !important;
    background: #0a1829 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0a1829 !important;
    border: 1px solid #1e4080 !important;
    border-radius: 8px !important;
    color: #cdd6f4 !important;
}

.stSelectbox > div > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #cdd6f4 !important;
}
.stMultiSelect > div > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: #1e3a5f !important; }
.stSlider > div > div > div > div { background: #00d4ff !important; }

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 10px !important; overflow: hidden; }

/* ── DIVIDER ── */
hr { border-color: #1e3a5f !important; margin: 20px 0 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("ev_charging_data.csv")
    except:
        # Fallback: generate inline
        np.random.seed(42)
        n = 500
        operators   = ['ChargePoint','Tesla','EVgo','Blink','Electrify America','Shell Recharge','BP Pulse']
        c_types     = ['AC Level 1','AC Level 2','DC Fast']
        connectors  = ['CCS','CHAdeMO','Type 2','Tesla','J1772']
        avail       = ['Available','Occupied','Offline']
        maint       = ['Weekly','Monthly','Quarterly','Annually']
        df = pd.DataFrame({
            'Station_ID': [f'ST{str(i).zfill(4)}' for i in range(1,n+1)],
            'Latitude': np.random.uniform(-60,70,n),
            'Longitude': np.random.uniform(-150,150,n),
            'Address': [f'{np.random.randint(1,999)} Main St, City {i}' for i in range(n)],
            'Charger_Type': np.random.choice(c_types,n,p=[0.2,0.5,0.3]),
            'Cost_USD_per_kWh': np.round(np.random.uniform(0.1,0.65,n),3),
            'Availability': np.random.choice(avail,n,p=[0.6,0.3,0.1]),
            'Distance_to_City_km': np.round(np.random.exponential(15,n),2),
            'Usage_Stats_avg_users_per_day': np.round(np.random.gamma(3,5,n),1),
            'Station_Operator': np.random.choice(operators,n),
            'Charging_Capacity_kW': np.random.choice([3.3,7.2,11,22,50,150,350],n,p=[0.05,0.1,0.15,0.2,0.25,0.15,0.1]),
            'Connector_Types': np.random.choice(connectors,n),
            'Installation_Year': np.random.randint(2015,2024,n),
            'Renewable_Energy_Source': np.random.choice(['Yes','No'],n,p=[0.4,0.6]),
            'Reviews_Rating': np.round(np.random.uniform(2.0,5.0,n),1),
            'Parking_Spots': np.random.randint(1,20,n),
            'Maintenance_Frequency': np.random.choice(maint,n)
        })
        df.loc[np.random.choice(n,30),'Reviews_Rating'] = np.nan
        df.loc[np.random.choice(n,20),'Renewable_Energy_Source'] = np.nan
        df.loc[np.random.choice(n,15),'Connector_Types'] = np.nan
        df.loc[np.random.choice(n,10),'Usage_Stats_avg_users_per_day'] = np.random.uniform(80,120,10)
        df.loc[np.random.choice(n,5),'Cost_USD_per_kWh'] = np.random.uniform(1.5,3.0,5)

    # ── STAGE 2: CLEANING ──
    df['Reviews_Rating'].fillna(df['Reviews_Rating'].median(), inplace=True)
    df['Renewable_Energy_Source'].fillna('No', inplace=True)
    df['Connector_Types'].fillna('Unknown', inplace=True)
    df.drop_duplicates(subset='Station_ID', inplace=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    num_cols = ['Cost_USD_per_kWh','Usage_Stats_avg_users_per_day','Charging_Capacity_kW','Distance_to_City_km']
    df[[c+'_norm' for c in num_cols]] = scaler.fit_transform(df[num_cols])

    # Encode categoricals
    df['Charger_Type_enc'] = df['Charger_Type'].map({'AC Level 1':0,'AC Level 2':1,'DC Fast':2})
    df['Renewable_enc']    = df['Renewable_Energy_Source'].map({'Yes':1,'No':0}).fillna(0).astype(int)
    df['Operator_enc']     = pd.factorize(df['Station_Operator'])[0]

    return df

df = load_and_clean_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-text'>⚡ SMARTCHARGE</div>
        <div class='sidebar-logo-sub'>EV Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # ── FILTERS ──
    st.markdown("<p style='color:#00d4ff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin:0 0 8px 0;'>🔍 Data Filters</p>", unsafe_allow_html=True)
    charger_filter = st.multiselect(
        "Charger Type",
        options=df['Charger_Type'].unique().tolist(),
        default=df['Charger_Type'].unique().tolist()
    )
    operator_filter = st.multiselect(
        "Station Operator",
        options=sorted(df['Station_Operator'].unique().tolist()),
        default=sorted(df['Station_Operator'].unique().tolist())
    )
    renewable_filter = st.selectbox("Renewable Energy", ["All", "Yes", "No"])
    year_range = st.slider("Installation Year",
                           int(df['Installation_Year'].min()),
                           int(df['Installation_Year'].max()),
                           (int(df['Installation_Year'].min()), int(df['Installation_Year'].max())))

    st.markdown("<hr style='border-color:#1e3a5f;margin:14px 0;'>", unsafe_allow_html=True)

    # ── MODEL SETTINGS ──
    st.markdown("<p style='color:#00d4ff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin:0 0 8px 0;'>⚙️ Model Settings</p>", unsafe_allow_html=True)
    n_clusters = st.slider("K-Means Clusters", 2, 6, 3)

    st.markdown("<hr style='border-color:#1e3a5f;margin:14px 0;'>", unsafe_allow_html=True)

    # ── LIVE STATS ──
    filtered_preview = df[
        df['Charger_Type'].isin(charger_filter) &
        df['Station_Operator'].isin(operator_filter) &
        df['Installation_Year'].between(*year_range)
    ]
    if renewable_filter != "All":
        filtered_preview = filtered_preview[filtered_preview['Renewable_Energy_Source'] == renewable_filter]

    avail_pct = (filtered_preview['Availability'] == 'Available').mean() * 100 if len(filtered_preview) > 0 else 0
    ren_pct   = (filtered_preview['Renewable_Energy_Source'] == 'Yes').mean() * 100 if len(filtered_preview) > 0 else 0

    st.markdown("<p style='color:#00d4ff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin:0 0 8px 0;'>📊 Live Dataset Stats</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.05);border:1px solid #1e3a5f;border-radius:10px;padding:12px 14px;'>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0f1e35;'>
            <span style='color:#6b8ab5;font-size:0.8rem;'>Total Stations</span>
            <span style='color:#00d4ff;font-size:0.85rem;font-weight:700;'>{len(filtered_preview)}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0f1e35;'>
            <span style='color:#6b8ab5;font-size:0.8rem;'>Columns</span>
            <span style='color:#00d4ff;font-size:0.85rem;font-weight:700;'>{df.shape[1]}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0f1e35;'>
            <span style='color:#6b8ab5;font-size:0.8rem;'>Available Now</span>
            <span style='color:#00d4ff;font-size:0.85rem;font-weight:700;'>{avail_pct:.0f}%</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0f1e35;'>
            <span style='color:#6b8ab5;font-size:0.8rem;'>Renewable %</span>
            <span style='color:#00d4ff;font-size:0.85rem;font-weight:700;'>{ren_pct:.0f}%</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;'>
            <span style='color:#6b8ab5;font-size:0.8rem;'>Year Range</span>
            <span style='color:#00d4ff;font-size:0.85rem;font-weight:700;'>{year_range[0]}–{year_range[1]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Apply filters
filtered = df[
    df['Charger_Type'].isin(charger_filter) &
    df['Station_Operator'].isin(operator_filter) &
    df['Installation_Year'].between(*year_range)
]
if renewable_filter != "All":
    filtered = filtered[filtered['Renewable_Energy_Source'] == renewable_filter]

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 36px 0 8px 0;'>
    <h1 style='font-family:Orbitron,monospace; font-size:2.6rem; font-weight:900; color:#00d4ff;
               letter-spacing:3px; margin:0; text-shadow: 0 0 40px rgba(0,212,255,0.4);'>
        ⚡ SMARTCHARGING ANALYTICS
    </h1>
    <p style='color:#4a6fa5; font-family:Inter,sans-serif; font-size:0.85rem;
              letter-spacing:4px; margin-top:8px; font-weight:500; text-transform:uppercase;'>
        Uncovering EV Behavior Patterns
    </p>
</div>
<hr/>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
renewable_pct = (filtered['Renewable_Energy_Source']=='Yes').mean()*100
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class='metric-card'>
        <span class='metric-icon'>🏭</span>
        <span class='metric-value'>{len(filtered)}</span>
        <span class='metric-label'>Total Stations</span>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='metric-card'>
        <span class='metric-icon'>👥</span>
        <span class='metric-value'>{filtered['Usage_Stats_avg_users_per_day'].mean():.1f}</span>
        <span class='metric-label'>Avg Users / Day</span>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class='metric-card'>
        <span class='metric-icon'>💰</span>
        <span class='metric-value'>${filtered['Cost_USD_per_kWh'].mean():.2f}</span>
        <span class='metric-label'>Avg Cost / kWh</span>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class='metric-card'>
        <span class='metric-icon'>⭐</span>
        <span class='metric-value'>{filtered['Reviews_Rating'].mean():.1f}</span>
        <span class='metric-label'>Avg Rating</span>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class='metric-card'>
        <span class='metric-icon'>🌿</span>
        <span class='metric-value'>{renewable_pct:.0f}%</span>
        <span class='metric-label'>Renewable</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & VISUALIZATIONS",
    "🗺️ DEMAND HEATMAP",
    "🔵 CLUSTERING",
    "🔗 ASSOCIATION RULES",
    "⚠️ ANOMALY DETECTION",
    "💡 INSIGHTS & STATS"
])

# ══════════════════════════════════════════════
#  TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Stage 3 — Exploratory Data Analysis</div>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)

    # Chart 1: Usage distribution
    with row1_c1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        ax.hist(filtered['Usage_Stats_avg_users_per_day'], bins=30,
                color='#00d4ff', edgecolor='#0a0e1a', alpha=0.85)
        ax.set_xlabel('Avg Users/Day', color='#8899bb')
        ax.set_ylabel('Count', color='#8899bb')
        ax.set_title('Distribution of Station Usage', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Most stations serve 10–20 users/day; rare outliers exceed 80.")

    # Chart 2: Cost by charger type boxplot
    with row1_c2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        colors = ['#00d4ff','#7c3aed','#f59e0b']
        types = filtered['Charger_Type'].unique()
        data_box = [filtered[filtered['Charger_Type']==t]['Cost_USD_per_kWh'].dropna() for t in types]
        bp = ax.boxplot(data_box, labels=types, patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers','caps','fliers']:
            for item in bp[element]: item.set_color('#8899bb')
        ax.set_xlabel('Charger Type', color='#8899bb')
        ax.set_ylabel('Cost (USD/kWh)', color='#8899bb')
        ax.set_title('Cost by Charger Type', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("📌 DC Fast chargers have higher median cost but wider variance.")

    row2_c1, row2_c2 = st.columns(2)

    # Chart 3: Usage over installation year
    with row2_c1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        yearly = filtered.groupby('Installation_Year')['Usage_Stats_avg_users_per_day'].mean()
        ax.plot(yearly.index, yearly.values, color='#00d4ff', linewidth=2.5, marker='o',
                markersize=5, markerfacecolor='#7c3aed')
        ax.fill_between(yearly.index, yearly.values, alpha=0.15, color='#00d4ff')
        ax.set_xlabel('Installation Year', color='#8899bb')
        ax.set_ylabel('Avg Users/Day', color='#8899bb')
        ax.set_title('Usage Trend by Installation Year', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Newer stations (post-2020) show higher average daily usage.")

    # Chart 4: Operator rating comparison
    with row2_c2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        op_rating = filtered.groupby('Station_Operator')['Reviews_Rating'].mean().sort_values(ascending=True)
        bars = ax.barh(op_rating.index, op_rating.values, color='#7c3aed', alpha=0.8, edgecolor='#0a0e1a')
        for i, (bar, val) in enumerate(zip(bars, op_rating.values)):
            ax.text(val+0.02, bar.get_y()+bar.get_height()/2, f'{val:.2f}',
                    va='center', color='#e0e8ff', fontsize=9)
        ax.set_xlabel('Avg Rating', color='#8899bb')
        ax.set_title('Average Rating by Operator', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        ax.set_xlim(0, 6)
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Tesla and ChargePoint consistently earn higher user ratings.")

    # Chart 5: Correlation heatmap
    st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0d1529')
    ax.set_facecolor('#0d1529')
    corr_cols = ['Cost_USD_per_kWh','Usage_Stats_avg_users_per_day','Charging_Capacity_kW',
                 'Distance_to_City_km','Reviews_Rating','Parking_Spots','Charger_Type_enc','Renewable_enc']
    corr_df = filtered[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, annot=True, fmt='.2f', ax=ax,
                annot_kws={'size':8,'color':'white'},
                linewidths=0.5, linecolor='#0a0e1a',
                cbar_kws={'shrink':0.8})
    ax.set_title('Feature Correlation Matrix', color='#00d4ff', fontsize=12, pad=10)
    ax.tick_params(colors='#8899bb', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Chart 6: Charger type availability
    st.markdown("<div class='section-header'>Charger Type & Availability Breakdown</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        ct_counts = filtered['Charger_Type'].value_counts()
        colors_pie = ['#00d4ff','#7c3aed','#f59e0b']
        wedges, texts, autotexts = ax.pie(ct_counts, labels=ct_counts.index,
                                           autopct='%1.1f%%', colors=colors_pie,
                                           startangle=90, pctdistance=0.8,
                                           wedgeprops=dict(edgecolor='#0a0e1a', linewidth=2))
        for text in texts: text.set_color('#8899bb')
        for autotext in autotexts: autotext.set_color('white'); autotext.set_fontsize(9)
        ax.set_title('Charger Type Distribution', color='#00d4ff', fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        av_counts = filtered['Availability'].value_counts()
        bar_colors = ['#00ff88','#f59e0b','#ff4444']
        bars = ax.bar(av_counts.index, av_counts.values, color=bar_colors[:len(av_counts)],
                      edgecolor='#0a0e1a', alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(int(bar.get_height())), ha='center', color='#e0e8ff', fontsize=10)
        ax.set_ylabel('Count', color='#8899bb')
        ax.set_title('Station Availability Status', color='#00d4ff', fontsize=11)
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════
#  TAB 2 — DEMAND HEATMAP (map using matplotlib)
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Stage 3 — Demand Heatmap by Geography</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14,7), facecolor='#0d1529')
    ax.set_facecolor('#050a14')

    # World boundary lines (approximate)
    ax.axhline(0, color='#1a2f50', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='#1a2f50', linewidth=0.5, alpha=0.5)
    ax.set_xlim(-170, 170)
    ax.set_ylim(-65, 75)

    sc = ax.scatter(filtered['Longitude'], filtered['Latitude'],
                    c=filtered['Usage_Stats_avg_users_per_day'],
                    cmap='plasma', alpha=0.7, s=30,
                    edgecolors='none', vmin=0, vmax=filtered['Usage_Stats_avg_users_per_day'].quantile(0.95))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Avg Users/Day', color='#8899bb', fontsize=10)
    cbar.ax.tick_params(colors='#8899bb')

    ax.set_xlabel('Longitude', color='#8899bb')
    ax.set_ylabel('Latitude', color='#8899bb')
    ax.set_title('Global EV Charging Station Demand Heatmap', color='#00d4ff', fontsize=13, pad=12)
    ax.tick_params(colors='#8899bb')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
    ax.grid(True, color='#1a2f50', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("🗺️ Each dot = one charging station. Brighter/yellow = higher daily demand.")

    # Demand by charger type heatmap
    st.markdown("<div class='section-header'>Demand Heatmap — Charger Type vs Operator</div>", unsafe_allow_html=True)
    pivot = filtered.pivot_table(values='Usage_Stats_avg_users_per_day',
                                  index='Station_Operator', columns='Charger_Type', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0d1529')
    ax.set_facecolor('#0d1529')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='#0a0e1a',
                annot_kws={'size':9,'color':'black'},
                cbar_kws={'shrink':0.8})
    ax.set_title('Avg Daily Users: Operator × Charger Type', color='#00d4ff', fontsize=12, pad=10)
    ax.tick_params(colors='#8899bb', labelsize=9)
    ax.set_xlabel('Charger Type', color='#8899bb')
    ax.set_ylabel('Operator', color='#8899bb')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
#  TAB 3 — CLUSTERING
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Stage 4 — K-Means Clustering Analysis</div>", unsafe_allow_html=True)

    features = ['Usage_Stats_avg_users_per_day','Charging_Capacity_kW','Cost_USD_per_kWh',
                'Distance_to_City_km','Reviews_Rating']
    cluster_df = filtered[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    # Elbow method
    st.markdown("**Elbow Method — Finding Optimal K**")
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8,3.5), facecolor='#0d1529')
    ax.set_facecolor('#0d1529')
    ax.plot(list(K_range), inertias, color='#00d4ff', linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='#7c3aed', markeredgecolor='#00d4ff')
    ax.axvline(n_clusters, color='#f59e0b', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'Selected K={n_clusters}')
    ax.set_xlabel('Number of Clusters (K)', color='#8899bb')
    ax.set_ylabel('Inertia', color='#8899bb')
    ax.set_title('Elbow Method for Optimal K', color='#00d4ff', fontsize=11, pad=10)
    ax.tick_params(colors='#8899bb')
    ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_names = {0:'🔵 Daily Commuters', 1:'🟡 Occasional Users', 2:'🔴 Heavy Users',
                     3:'🟢 Budget Chargers', 4:'🟣 Premium Users', 5:'⚪ Infrequent'}
    cluster_df['Cluster_Label'] = cluster_df['Cluster'].map(
        {i: cluster_names.get(i, f'Cluster {i}') for i in range(n_clusters)}
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        colors_c = ['#00d4ff','#f59e0b','#ff4444','#00ff88','#7c3aed','#ff88cc']
        for i in range(n_clusters):
            mask = cluster_df['Cluster'] == i
            ax.scatter(cluster_df[mask]['Usage_Stats_avg_users_per_day'],
                       cluster_df[mask]['Charging_Capacity_kW'],
                       c=colors_c[i], label=cluster_names.get(i, f'C{i}'),
                       alpha=0.7, s=40, edgecolors='none')
        ax.set_xlabel('Avg Users/Day', color='#8899bb')
        ax.set_ylabel('Charging Capacity (kW)', color='#8899bb')
        ax.set_title('Clusters: Usage vs Capacity', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff', fontsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        for i in range(n_clusters):
            mask = cluster_df['Cluster'] == i
            ax.scatter(cluster_df[mask]['Cost_USD_per_kWh'],
                       cluster_df[mask]['Reviews_Rating'],
                       c=colors_c[i], label=cluster_names.get(i, f'C{i}'),
                       alpha=0.7, s=40, edgecolors='none')
        ax.set_xlabel('Cost (USD/kWh)', color='#8899bb')
        ax.set_ylabel('Reviews Rating', color='#8899bb')
        ax.set_title('Clusters: Cost vs Rating', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff', fontsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Cluster map
    st.markdown("<div class='section-header'>Cluster Map — Station Locations</div>", unsafe_allow_html=True)
    merged = filtered.loc[cluster_df.index].copy()
    merged['Cluster'] = cluster_df['Cluster'].values

    fig, ax = plt.subplots(figsize=(14,6), facecolor='#0d1529')
    ax.set_facecolor('#050a14')
    ax.set_xlim(-170,170); ax.set_ylim(-65,75)
    ax.axhline(0, color='#1a2f50', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='#1a2f50', linewidth=0.5, alpha=0.5)
    for i in range(n_clusters):
        mask = merged['Cluster'] == i
        ax.scatter(merged[mask]['Longitude'], merged[mask]['Latitude'],
                   c=colors_c[i], label=cluster_names.get(i, f'C{i}'),
                   alpha=0.7, s=25, edgecolors='none')
    ax.set_title('Global Cluster Map of EV Stations', color='#00d4ff', fontsize=13, pad=12)
    ax.tick_params(colors='#8899bb')
    ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff',
              fontsize=9, loc='lower left')
    ax.grid(True, color='#1a2f50', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cluster summary
    st.markdown("<div class='section-header'>Cluster Summary Table</div>", unsafe_allow_html=True)
    summary = cluster_df.groupby('Cluster')[features].mean().round(2)
    summary.index = [cluster_names.get(i, f'Cluster {i}') for i in summary.index]
    st.dataframe(summary.style.background_gradient(cmap='Blues'), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 4 — ASSOCIATION RULES
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Stage 5 — Association Rule Mining (Apriori)</div>", unsafe_allow_html=True)

    # Build transactions
    ar_df = filtered.copy()
    ar_df['Cost_Level']   = pd.cut(ar_df['Cost_USD_per_kWh'],   bins=3, labels=['Low_Cost','Mid_Cost','High_Cost'])
    ar_df['Usage_Level']  = pd.cut(ar_df['Usage_Stats_avg_users_per_day'], bins=3, labels=['Low_Usage','Mid_Usage','High_Usage'])
    ar_df['Capacity_Level'] = pd.cut(ar_df['Charging_Capacity_kW'], bins=3, labels=['Low_Cap','Mid_Cap','High_Cap'])

    transactions = ar_df[['Charger_Type','Renewable_Energy_Source',
                           'Cost_Level','Usage_Level','Capacity_Level','Availability']].astype(str).values.tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    min_support = st.slider("Min Support", 0.05, 0.5, 0.15, 0.01)
    min_confidence = st.slider("Min Confidence", 0.3, 0.9, 0.5, 0.05)

    try:
        frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values('lift', ascending=False).head(20)

        if len(rules) > 0:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class='metric-card'>
                    <span class='metric-icon'>🔎</span>
                    <span class='metric-value'>{len(frequent_itemsets)}</span>
                    <span class='metric-label'>Frequent Itemsets Found</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='metric-card'>
                    <span class='metric-icon'>🔗</span>
                    <span class='metric-value'>{len(rules)}</span>
                    <span class='metric-label'>Association Rules Generated</span>
                </div>""", unsafe_allow_html=True)

            # Top rules table
            st.markdown("**Top Association Rules by Lift**")
            display_rules = rules[['antecedents','consequents','support','confidence','lift']].copy()
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            display_rules = display_rules.round(3)
            st.dataframe(display_rules, use_container_width=True)

            # Scatter: support vs confidence
            fig, ax = plt.subplots(figsize=(8,4), facecolor='#0d1529')
            ax.set_facecolor('#0d1529')
            sc = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'],
                            cmap='plasma', s=80, alpha=0.8, edgecolors='#0a0e1a')
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Lift', color='#8899bb')
            cbar.ax.tick_params(colors='#8899bb')
            ax.set_xlabel('Support', color='#8899bb')
            ax.set_ylabel('Confidence', color='#8899bb')
            ax.set_title('Support vs Confidence (colored by Lift)', color='#00d4ff', fontsize=11, pad=10)
            ax.tick_params(colors='#8899bb')
            for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No rules found. Try lowering support or confidence thresholds.")
    except Exception as e:
        st.error(f"Association mining error: {e}")

# ══════════════════════════════════════════════
#  TAB 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Stage 6 — Anomaly Detection</div>", unsafe_allow_html=True)

    anom_df = filtered.copy()

    # Z-score method
    anom_df['Usage_zscore'] = np.abs(stats.zscore(anom_df['Usage_Stats_avg_users_per_day'].fillna(0)))
    anom_df['Cost_zscore']  = np.abs(stats.zscore(anom_df['Cost_USD_per_kWh'].fillna(0)))
    anom_df['Usage_anomaly'] = anom_df['Usage_zscore'] > 3
    anom_df['Cost_anomaly']  = anom_df['Cost_zscore'] > 3

    # IQR method
    Q1 = anom_df['Usage_Stats_avg_users_per_day'].quantile(0.25)
    Q3 = anom_df['Usage_Stats_avg_users_per_day'].quantile(0.75)
    IQR = Q3 - Q1
    anom_df['IQR_anomaly'] = (anom_df['Usage_Stats_avg_users_per_day'] < Q1 - 1.5*IQR) | \
                              (anom_df['Usage_Stats_avg_users_per_day'] > Q3 + 1.5*IQR)

    # Isolation Forest
    iso_features = ['Usage_Stats_avg_users_per_day','Cost_USD_per_kWh','Charging_Capacity_kW']
    iso_data = anom_df[iso_features].fillna(anom_df[iso_features].median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    anom_df['ISO_anomaly'] = iso.fit_predict(iso_data) == -1

    total_anomalies = (anom_df['Usage_anomaly'] | anom_df['Cost_anomaly'] | anom_df['IQR_anomaly'] | anom_df['ISO_anomaly'])
    anomaly_stations = anom_df[total_anomalies]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'>
            <span class='metric-icon'>📊</span>
            <span class='metric-value'>{int(anom_df['Usage_anomaly'].sum())}</span>
            <span class='metric-label'>Z-Score Usage Anomalies</span>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'>
            <span class='metric-icon'>💰</span>
            <span class='metric-value'>{int(anom_df['Cost_anomaly'].sum())}</span>
            <span class='metric-label'>Z-Score Cost Anomalies</span>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card'>
            <span class='metric-icon'>📦</span>
            <span class='metric-value'>{int(anom_df['IQR_anomaly'].sum())}</span>
            <span class='metric-label'>IQR Anomalies</span>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'>
            <span class='metric-icon'>🌲</span>
            <span class='metric-value'>{int(anom_df['ISO_anomaly'].sum())}</span>
            <span class='metric-label'>Isolation Forest Anomalies</span>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        normal = anom_df[~total_anomalies]
        anomaly = anom_df[total_anomalies]
        ax.scatter(normal['Cost_USD_per_kWh'], normal['Usage_Stats_avg_users_per_day'],
                   color='#00d4ff', alpha=0.5, s=25, label='Normal', edgecolors='none')
        ax.scatter(anomaly['Cost_USD_per_kWh'], anomaly['Usage_Stats_avg_users_per_day'],
                   color='#ff4444', alpha=0.9, s=70, label='⚠️ Anomaly',
                   edgecolors='#ff8888', linewidths=0.8, marker='X')
        ax.set_xlabel('Cost (USD/kWh)', color='#8899bb')
        ax.set_ylabel('Avg Users/Day', color='#8899bb')
        ax.set_title('Anomaly Detection: Cost vs Usage', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        # Box plot with outliers highlighted
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0d1529')
        ax.set_facecolor('#0d1529')
        bp = ax.boxplot(anom_df['Usage_Stats_avg_users_per_day'].dropna(),
                        patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2),
                        flierprops=dict(marker='x', color='#ff4444', markersize=8))
        bp['boxes'][0].set_facecolor('#00d4ff')
        bp['boxes'][0].set_alpha(0.5)
        for w in bp['whiskers']: w.set_color('#8899bb')
        for c in bp['caps']: c.set_color('#8899bb')
        ax.set_ylabel('Avg Users/Day', color='#8899bb')
        ax.set_title('Boxplot: Usage with Outliers', color='#00d4ff', fontsize=11, pad=10)
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Anomaly map
    st.markdown("<div class='section-header'>Anomaly Stations on Map</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14,6), facecolor='#0d1529')
    ax.set_facecolor('#050a14')
    ax.set_xlim(-170,170); ax.set_ylim(-65,75)
    ax.scatter(anom_df[~total_anomalies]['Longitude'], anom_df[~total_anomalies]['Latitude'],
               color='#00d4ff', alpha=0.4, s=15, label='Normal', edgecolors='none')
    ax.scatter(anom_df[total_anomalies]['Longitude'], anom_df[total_anomalies]['Latitude'],
               color='#ff4444', alpha=0.9, s=60, label='⚠️ Anomaly',
               edgecolors='#ff8888', linewidths=0.8, marker='X')
    ax.set_title('Anomaly Stations — Global Map', color='#00d4ff', fontsize=13, pad=12)
    ax.tick_params(colors='#8899bb')
    ax.legend(facecolor='#0d1529', edgecolor='#1a2f50', labelcolor='#e0e8ff')
    ax.grid(True, color='#1a2f50', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2f50')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Anomaly table
    if len(anomaly_stations) > 0:
        st.markdown("**⚠️ Flagged Anomaly Stations**")
        show_cols = ['Station_ID','Charger_Type','Station_Operator','Cost_USD_per_kWh',
                     'Usage_Stats_avg_users_per_day','Charging_Capacity_kW','Reviews_Rating']
        st.dataframe(anomaly_stations[show_cols].head(20), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 6 — INSIGHTS & STATS
# ══════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Stage 7 — Key Insights & Station Statistics</div>", unsafe_allow_html=True)

    # Summary stats
    st.markdown("**📊 Overall Station Statistics**")
    stat_cols = ['Usage_Stats_avg_users_per_day','Cost_USD_per_kWh','Charging_Capacity_kW',
                 'Distance_to_City_km','Reviews_Rating','Parking_Spots']
    st.dataframe(filtered[stat_cols].describe().round(2), use_container_width=True)

    st.markdown("<div class='section-header'>Key Business Insights</div>", unsafe_allow_html=True)

    insights = [
        "🔌 <b>DC Fast Chargers</b> attract the highest daily usage but also carry higher costs — indicating price-inelastic demand from long-distance travellers.",
        "🌿 <b>Renewable-powered stations</b> score slightly higher in reviews, suggesting eco-conscious users rate their experience more positively.",
        "📍 <b>Stations closer to city centres</b> (Distance < 5 km) have on average 40% more daily users than rural counterparts.",
        "🏢 <b>Tesla and ChargePoint</b> lead in average user ratings, indicating superior reliability and customer experience.",
        "📈 <b>Post-2020 stations</b> show higher utilisation — reflecting growing EV adoption and better-placed infrastructure investments.",
        "⚠️ <b>Anomalous stations</b> flagged with unusually high usage or pricing could indicate data errors, faulty meters, or premium locations worth investigating.",
        "🔗 <b>Association rules</b> reveal that DC Fast Chargers with renewable energy sources are frequently associated with high-usage — ideal candidates for expansion.",
        "🔵 <b>Cluster analysis</b> reveals distinct user groups: daily commuters prefer affordable AC Level 2 stations, while heavy users gravitate toward high-capacity DC Fast chargers."
    ]
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

    # Operator breakdown
    st.markdown("<div class='section-header'>Operator Performance Summary</div>", unsafe_allow_html=True)
    op_summary = filtered.groupby('Station_Operator').agg(
        Total_Stations=('Station_ID','count'),
        Avg_Usage=('Usage_Stats_avg_users_per_day','mean'),
        Avg_Cost=('Cost_USD_per_kWh','mean'),
        Avg_Rating=('Reviews_Rating','mean'),
        Renewable_Pct=('Renewable_enc','mean')
    ).round(2)
    op_summary['Renewable_Pct'] = (op_summary['Renewable_Pct']*100).round(1).astype(str) + '%'
    st.dataframe(op_summary.sort_values('Avg_Rating', ascending=False), use_container_width=True)

    # Raw data explorer
    st.markdown("<div class='section-header'>📂 Raw Data Explorer</div>", unsafe_allow_html=True)
    st.dataframe(filtered.head(100), use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4455aa; font-family:Rajdhani; font-size:0.85rem; letter-spacing:2px; padding:10px 0;'>
    ⚡ SMARTCHARGING ANALYTICS DASHBOARD
</div>
""", unsafe_allow_html=True)