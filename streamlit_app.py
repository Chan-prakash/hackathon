import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Ghana Healthcare Coverage · Virtue Foundation",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #F7F8FA; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .app-header { padding: 1.5rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.5rem; }
    .app-header h1 { font-size: 1.6rem; font-weight: 700; color: #1A2332; margin: 0; letter-spacing: -0.02em; }
    .app-header p  { font-size: 0.9rem; color: #5A6B7F; margin: 0.25rem 0 0 0; }
    .metric-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.25rem 1.5rem; transition: box-shadow 0.2s ease; }
    .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    .metric-card .label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #5A6B7F; margin-bottom: 0.35rem; }
    .metric-card .value  { font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
    .metric-card .subtext { font-size: 0.75rem; color: #8896A6; margin-top: 0.2rem; }
    .metric-card.teal .value { color: #0D7377; }
    .metric-card.red   .value { color: #D64545; }
    .metric-card.amber .value { color: #E8871E; }
    .metric-card.green .value { color: #2D8B4E; }
    .result-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; transition: box-shadow 0.2s ease; }
    .result-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .result-card .hospital-name  { font-size: 1rem; font-weight: 600; color: #1A2332; }
    .result-card .hospital-meta  { font-size: 0.8rem; color: #5A6B7F; margin-top: 0.2rem; }
    .result-card .hospital-desc  { font-size: 0.85rem; color: #3D4F5F; margin-top: 0.5rem; line-height: 1.5; }
    .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; }
    .badge-high     { background: #D1FAE5; color: #065F46; }
    .badge-medium   { background: #FEF3C7; color: #92400E; }
    .badge-low      { background: #FEE2E2; color: #991B1B; }
    .badge-critical { background: #FEE2E2; color: #991B1B; }
    .badge-high-risk { background: #FFEDD5; color: #9A3412; }
    .badge-moderate  { background: #FEF9C3; color: #854D0E; }
    .badge-adequate  { background: #D1FAE5; color: #065F46; }
    .risk-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .risk-critical { background-color: #D64545; }
    .risk-high     { background-color: #E8871E; }
    .risk-moderate { background-color: #D4A72C; }
    .risk-adequate { background-color: #2D8B4E; }
    .ai-answer { background: #F0FDFA; border: 1px solid #99F6E4; border-radius: 10px; padding: 1.25rem 1.5rem; margin-top: 1rem; font-size: 0.9rem; line-height: 1.6; color: #1A2332; }
    .ai-answer .answer-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #0D7377; margin-bottom: 0.5rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1A2332; margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #0D7377; display: inline-block; }
    .gap-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; }
    .gap-table th { background: #F1F5F9; color: #475569; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; padding: 0.75rem 1rem; text-align: left; border-bottom: 2px solid #E2E8F0; }
    .gap-table td { padding: 0.65rem 1rem; border-bottom: 1px solid #F1F5F9; color: #334155; }
    .gap-table tr:hover td { background: #F8FAFC; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.9rem; color: #5A6B7F; padding: 0.75rem 1.25rem; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #0D7377 !important; border-bottom-color: #0D7377 !important; background: transparent !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {display: none;}
    .anomaly-flag { background: #FFFBEB; border-left: 3px solid #F59E0B; padding: 0.5rem 0.75rem; margin-top: 0.5rem; font-size: 0.8rem; color: #92400E; border-radius: 0 6px 6px 0; }
    div[data-testid="column"] > div { height: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and deduplicate hospital data."""
    df = None
    for path in ["data/hospital_metadata.csv", "hospital_metadata.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        st.error("Hospital data file not found. Place `hospital_metadata.csv` in `data/` folder.")
        st.stop()

    df = df.fillna("")

    # ── Deduplication ──
    def completeness_score(row):
        score = 0
        for col in ['description', 'procedure', 'equipment', 'capability', 'specialties', 'address_city']:
            val = str(row.get(col, '')).strip()
            if val not in ['', 'nan', '[]', "['']"]:
                score += 1
        return score

    df['_completeness'] = df.apply(completeness_score, axis=1)
    df = df.sort_values('_completeness', ascending=False).drop_duplicates(subset=['name'], keep='first')
    df = df.drop(columns=['_completeness']).reset_index(drop=True)

    # ── Normalize region names ──
    region_fixes = {
        'Accra East': 'Greater Accra', 'Accra North': 'Greater Accra',
        'Ga East Municipality': 'Greater Accra', 'Ga East Municipality, Greater Accra Region': 'Greater Accra',
        'Ledzokuku-Krowor': 'Greater Accra', 'Kpone Katamanso': 'Greater Accra',
        'Shai Osudoku District, Greater Accra Region': 'Greater Accra', 'Tema West Municipal': 'Greater Accra',
        'Asokwa-Kumasi': 'Ashanti', 'Ejisu Municipal': 'Ashanti',
        'Ahafo Ano South-East': 'Ashanti', 'SH': 'Ashanti',
        'KEEA': 'Central', 'Central Ghana': 'Central', 'Central Tongu District': 'Volta',
        'Asutifi South': 'Ahafo', 'Bono': 'Brong Ahafo', 'Dormaa East': 'Brong Ahafo',
        'Techiman Municipal': 'Brong Ahafo', 'Takoradi': 'Western',
        'Sissala West District': 'Upper West', 'Ghana': 'Unknown',
    }
    df['region_clean'] = df['region_clean'].replace(region_fixes)
    df.loc[df['region_clean'].isin(['', 'nan', 'Unknown']), 'region_clean'] = 'Unknown'

    # ── City-based region fix for remaining Unknowns ──
    city_to_region = {
        'accra': 'Greater Accra', 'tema': 'Greater Accra', 'madina': 'Greater Accra',
        'legon': 'Greater Accra', 'adenta': 'Greater Accra', 'dansoman': 'Greater Accra',
        'lapaz': 'Greater Accra', 'achimota': 'Greater Accra', 'nungua': 'Greater Accra',
        'weija': 'Greater Accra', 'ashaiman': 'Greater Accra', 'labadi': 'Greater Accra',
        'kumasi': 'Ashanti', 'obuasi': 'Ashanti', 'ejisu': 'Ashanti',
        'takoradi': 'Western', 'sekondi': 'Western', 'tarkwa': 'Western',
        'cape coast': 'Central', 'winneba': 'Central', 'saltpond': 'Central',
        'tamale': 'Northern', 'yendi': 'Northern', 'savelugu': 'Northern',
        'bolgatanga': 'Upper East', 'wa': 'Upper West', 'ho': 'Volta', 'sunyani': 'Brong Ahafo'
    }

    def fix_unknown_region(row):
        if row['region_clean'] != 'Unknown': return row['region_clean']
        city = str(row.get('address_city', '')).lower().strip()
        for city_key, region in city_to_region.items():
            if city_key in city: return region
        return 'Unknown'

    df['region_clean'] = df.apply(fix_unknown_region, axis=1)
    return df

# ─── NEW: Dynamic Region Stats ───────────────────────────────
def get_live_region_stats(df):
    """Calculates live hospital counts so the LLM never hallucinates."""
    stats = f"GHANA HOSPITAL DATABASE — LIVE STATISTICS ({len(df)} total hospitals):\n"
    counts = df['region_clean'].value_counts()
    for region, count in counts.items():
        if region != "Unknown":
            stats += f"- {region}: {count} hospitals\n"
    stats += f"- Unknown region: {counts.get('Unknown', 0)} hospitals (location data missing)\n"
    return stats

# ─── Gap Analysis ────────────────────────────────────────────
@st.cache_data
def build_gap_analysis(df):
    """Score each region on 8 critical medical services."""
    services = {
        'ICU':        ['icu', 'intensive care', 'critical care'],
        'Emergency':  ['emergency', 'accident', '24/7', '24 hour', '24-hour'],
        'Surgery':    ['surgery', 'surgical', 'operating theatre'],
        'Maternity':  ['maternity', 'obstetric', 'gynecology', 'delivery', 'gynaecology'],
        'Laboratory': ['laboratory', 'lab test', 'diagnostic lab'],
        'Imaging':    ['x-ray', 'xray', 'ultrasound', 'scan', 'mri', 'ct scan', 'radiology'],
        'Pediatrics': ['pediatric', 'paediatric', 'children', 'child care', 'neonatal'],
        'Pharmacy':   ['pharmacy', 'pharmaceutical', 'dispensary'],
    }

    def build_text(row):
        parts = []
        for col in ['description', 'procedure', 'equipment', 'capability', 'specialties']:
            val = str(row.get(col, ''))
            if val not in ['', 'nan', '[]', "['']"]: parts.append(val)
        return ' '.join(parts).lower()

    df_copy = df.copy()
    df_copy['_all_text'] = df_copy.apply(build_text, axis=1)

    for svc_name, keywords in services.items():
        df_copy[f'has_{svc_name}'] = df_copy['_all_text'].apply(lambda t: any(k in t for k in keywords))

    svc_cols = [f'has_{s}' for s in services]
    region_scores = df_copy[df_copy['region_clean'] != 'Unknown'].groupby('region_clean').agg(
        total_facilities=('name', 'count'), **{col: (col, 'sum') for col in svc_cols}
    ).reset_index()

    region_scores['services_available'] = sum((region_scores[f'has_{s}'] > 0).astype(int) for s in services)

    def get_risk(row):
        s, f = row['services_available'], row['total_facilities']
        if s <= 2 or f <= 3: return 'Critical'
        elif s <= 4 or f <= 8: return 'High Risk'
        elif s <= 6 or f <= 20: return 'Moderate'
        return 'Adequate'

    region_scores['risk_level'] = region_scores.apply(get_risk, axis=1)
    return region_scores.sort_values('services_available'), services

# ─── NEW: FAISS + RAG Directly From Disk ─────────────────────
@st.cache_resource
def load_faiss():
    """Loads the pre-built FAISS index directly from the file."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    index_path = "data/ghana_hospitals_full.index" if os.path.exists("data/ghana_hospitals_full.index") else "ghana_hospitals_full.index"
    if not os.path.exists(index_path):
        st.error(f"FAISS index not found at {index_path}. Please add it to your folder.")
        st.stop()
        
    index = faiss.read_index(index_path)
    return embedder, index

def search_hospitals(query, df, embedder, index, top_k=20, allowed_indices=None):
    """Searches the index and fetches top 20 results."""
    q_emb = embedder.encode([query]).astype('float32')
    
    # If we have filters, search the entire index to guarantee we find top_k matches post-filtering
    fetch_k = len(df) if allowed_indices is not None else top_k
    distances, indices = index.search(q_emb, min(fetch_k, len(df)))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if allowed_indices is not None and idx not in allowed_indices:
            continue
            
        row = df.iloc[idx]
        results.append({
            'name':        row['name'],
            'region':      row['region_clean'],
            'city':        str(row.get('address_city', '')),
            'type':        str(row.get('facilityTypeId', '')),
            'description': str(row.get('description', ''))[:300],
            'procedure':   str(row.get('procedure', '')),
            'capability':  str(row.get('capability', '')),
            'equipment':   str(row.get('equipment', '')),
            'specialties': str(row.get('specialties', '')),
            'confidence':  str(row.get('confidence', 'Medium')),
            'similarity':  round(1 / (1 + dist), 3),
        })
        if len(results) >= top_k:
            break
    return results

def detect_anomalies(result):
    anomalies = []
    ftype, all_text = str(result.get('type', '')).lower(), (result.get('procedure', '') + ' ' + result.get('capability', '') + ' ' + result.get('equipment', '')).lower()
    if ftype == 'clinic' and 'icu' in all_text: anomalies.append('Clinic claiming ICU capability — verify')
    if ftype == 'pharmacy' and 'surgery' in all_text: anomalies.append('Pharmacy claiming surgery — suspicious')
    if 'icu' in all_text and 'emergency' not in all_text: anomalies.append('Claims ICU but no emergency services listed')
    return anomalies

def get_llm_answer(question, results, groq_key, dynamic_stats):
    """Get AI answer using retrieved context and dynamic stats."""
    if not groq_key: return "Set the GROQ_KEY environment variable to enable AI answers."
    from groq import Groq
    client = Groq(api_key=groq_key)

    context = ""
    for i, r in enumerate(results, 1):
        context += f"Hospital {i}: {r['name']} | Region: {r['region']} | Type: {r['type']} | Capabilities: {r['capability'][:100]}\n"

    prompt = f"""You are a healthcare analyst for Virtue Foundation in Ghana.
You help NGO coordinators find hospitals and plan resource deployments.

{dynamic_stats}

Top {len(results)} relevant hospitals found by search:
{context}

Question: {question}

Instructions:
- For questions about HOW MANY hospitals or WHICH REGION HAS MOST, use the LIVE STATISTICS above.
- For questions about specific hospitals, use the search results.
- Keep answer under 200 words, factual, and helpful."""

    try:
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting AI response: {str(e)[:100]}"

# ─── Execution ───────────────────────────────────────────────
df = load_data()
gap_df, services_dict = build_gap_analysis(df)
dynamic_stats = get_live_region_stats(df)
embedder, index = load_faiss()
groq_key = os.environ.get("GROQ_KEY", "")

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    
    if st.button("↻ Reset Filters", use_container_width=True):
        st.session_state['sb_region'] = "All"
        st.session_state['sb_type'] = "All"
        st.session_state['sb_risk'] = "All"
        st.rerun()

    regions = sorted(df[df['region_clean'] != 'Unknown']['region_clean'].unique().tolist())
    selected_region = st.selectbox("Region", ["All"] + regions, index=0, key="sb_region")

    facility_types = sorted(df[df['facilityTypeId'] != '']['facilityTypeId'].unique().tolist())
    selected_type = st.selectbox("Facility type", ["All"] + facility_types, index=0, key="sb_type")
    
    selected_risk = st.selectbox("Risk level", ['All', 'Critical', 'High Risk', 'Moderate', 'Adequate'], index=0, key="sb_risk")

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**{len(df)}** unique facilities")
    st.markdown(f"**{df[df['region_clean'] != 'Unknown']['region_clean'].nunique()}** regions with data")
    st.markdown(f"**{len(gap_df[gap_df['risk_level'] == 'Critical'])}** critical desert regions")

filtered_df = df.copy()
if selected_region != "All": filtered_df = filtered_df[filtered_df['region_clean'] == selected_region]
if selected_type != "All": filtered_df = filtered_df[filtered_df['facilityTypeId'] == selected_type]

# ─── Header ──────────────────────────────────────────────────
st.markdown('<div class="app-header"><h1>⚕️ Ghana Healthcare Coverage</h1><p>Identify medical deserts and plan NGO resource deployment across Ghana\'s 16 regions</p></div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Search", "📋 Regional Analysis", "🗺️ Map", "🏥 Directory"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card teal"><div class="label">Total Facilities</div><div class="value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card red"><div class="label">Critical Deserts</div><div class="value">{len(gap_df[gap_df["risk_level"]=="Critical"])}</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card amber"><div class="label">Services Tracked</div><div class="value">{len(services_dict)}</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card green"><div class="label">Regions Covered</div><div class="value">{df[df["region_clean"]!="Unknown"]["region_clean"].nunique()}</div></div>', unsafe_allow_html=True)

    import plotly.express as px
    import plotly.graph_objects as go
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown('<div class="section-header">Facilities by Region</div>', unsafe_allow_html=True)
        rc = filtered_df[filtered_df['region_clean'] != 'Unknown'].groupby('region_clean')['name'].count().reset_index()
        rc.columns = ['Region', 'Count']
        rc = rc.sort_values('Count', ascending=True)
        risk_map = dict(zip(gap_df['region_clean'], gap_df['risk_level']))
        rc['Risk'] = rc['Region'].map(risk_map).fillna('Unknown')
        rc['Color'] = rc['Risk'].map({'Critical': '#D64545', 'High Risk': '#E8871E', 'Moderate': '#D4A72C', 'Adequate': '#2D8B4E', 'Unknown': '#94A3B8'})
        
        fig_bar = go.Figure(go.Bar(x=rc['Count'], y=rc['Region'], orientation='h', marker_color=rc['Color'], text=rc['Count'], textposition='outside'))
        fig_bar.update_layout(height=max(360, len(rc)*28), margin=dict(l=0,r=40,t=10,b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = gap_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig_donut = go.Figure(go.Pie(labels=risk_counts['Risk Level'], values=risk_counts['Count'], hole=0.55, marker_colors=[{'Critical': '#D64545', 'High Risk': '#E8871E', 'Moderate': '#D4A72C', 'Adequate': '#2D8B4E'}.get(r, '#94A3B8') for r in risk_counts['Risk Level']]))
        fig_donut.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ═══════════════════════════════════��══════════════════════════
with tab2:
    st.markdown('<div class="section-header">Search Healthcare Facilities</div>', unsafe_allow_html=True)
    query = st.text_input("Search", value=st.session_state.get('search_query', ''), placeholder="e.g. Which hospitals in Volta region have surgery?", label_visibility="collapsed")

    if query:
        st.session_state['search_query'] = query
        with st.spinner("Searching 987 hospitals..."):
            # Map sidebar filters directly into allowed indices for FAISS post-filtering
            allowed_indices = set(filtered_df.index) if (selected_region != "All" or selected_type != "All") else None
            results = [] if (allowed_indices is not None and len(allowed_indices) == 0) else search_hospitals(query, df, embedder, index, top_k=20, allowed_indices=allowed_indices)

        if not results:
            st.info("No facilities found matching your search and filter criteria.")
        else:
            if groq_key:
                with st.spinner("Generating analysis..."):
                    answer = get_llm_answer(query, results, groq_key, dynamic_stats)
                st.markdown(f'<div class="ai-answer"><div class="answer-label">🤖 AI Analysis</div>{answer}</div>', unsafe_allow_html=True)
    
            st.markdown(f'<div class="section-header">Matching Facilities ({len(results)})</div>', unsafe_allow_html=True)
            for r in results:
                anomalies = detect_anomalies(r)
                anomaly_html = "".join(f'<div class="anomaly-flag">⚠️ {a}</div>' for a in anomalies)
                desc_text = r['description'][:250] + "..." if len(r['description']) > 250 else r['description']
                desc_html = f'<div class="hospital-desc">{desc_text}</div>' if desc_text else ""
                st.markdown(f'<div class="result-card"><div class="hospital-name">{r["name"]} <span class="badge badge-{"high" if r["confidence"]=="High" else "medium" if r["confidence"]=="Medium" else "low"}">{r["confidence"]} confidence</span></div><div class="hospital-meta">📍 {r["region"]} · {r["city"]} · {r["type"]} · Match: {r["similarity"]:.1%}</div>{desc_html}{anomaly_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Regional Gap Analysis</div>', unsafe_allow_html=True)
    display_gap = gap_df[gap_df['risk_level'] == selected_risk] if selected_risk != "All" else gap_df

    table_html = '<table class="gap-table"><thead><tr><th>Region</th><th>Facilities</th><th>Services</th><th>Risk</th>' + "".join([f'<th>{s}</th>' for s in services_dict]) + '</tr></thead><tbody>'
    for _, row in display_gap.iterrows():
        risk_class = row['risk_level'].lower().replace(' ', '-')
        dot_class = f'risk-{risk_class}' if risk_class != 'high-risk' else 'risk-high'
        table_html += f'<tr><td><strong>{row["region_clean"]}</strong></td><td>{int(row["total_facilities"])}</td><td>{int(row["services_available"])}/8</td><td><span class="risk-dot {dot_class}"></span><span class="badge badge-{risk_class}">{row["risk_level"]}</span></td>'
        for svc in services_dict: table_html += f'<td style="color:{"#2D8B4E" if row.get(f"has_{svc}", 0) > 0 else "#D64545"};font-weight:600;text-align:center">{"✓" if row.get(f"has_{svc}", 0) > 0 else "✗"}</td>'
        table_html += '</tr>'
    st.markdown(table_html + '</tbody></table>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — MAP
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Medical Desert Map</div>', unsafe_allow_html=True)
    map_path = next((path for path in ["data/ghana_map.html", "ghana_map.html"] if os.path.exists(path)), None)
    if map_path:
        with open(map_path, 'r', encoding='utf-8') as f: st.components.v1.html(f.read(), height=600, scrolling=False)
    else:
        st.warning("Map file not found. Place `ghana_map.html` in the `data/` folder.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — HOSPITAL DIRECTORY
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Hospital Directory</div>', unsafe_allow_html=True)
    dir_search = st.text_input("Filter by name or city", placeholder="e.g. Korle Bu or Tamale")
    dir_df = filtered_df[filtered_df['name'].str.lower().str.contains(dir_search.lower()) | filtered_df['address_city'].str.lower().str.contains(dir_search.lower())] if dir_search else filtered_df
    
    display_cols = ['name', 'region_clean', 'address_city', 'facilityTypeId'] + [c for c in ['specialties', 'confidence'] if c in dir_df.columns]
    st.dataframe(dir_df[display_cols].rename(columns={'name': 'Hospital', 'region_clean': 'Region', 'address_city': 'City', 'facilityTypeId': 'Type', 'specialties': 'Specialties', 'confidence': 'Confidence'}), use_container_width=True, hide_index=True, height=500)