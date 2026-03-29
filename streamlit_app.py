import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
import json
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Ghana Healthcare Coverage · Virtue Foundation",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional CSS (Light Healthcare Theme) ──────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #F7F8FA;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1A2332;
    }

    /* ── Header ── */
    .app-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1A2332;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.9rem;
        color: #5A6B7F;
        margin: 0.25rem 0 0 0;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        flex: 1;
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #5A6B7F;
        margin-bottom: 0.35rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .subtext {
        font-size: 0.75rem;
        color: #8896A6;
        margin-top: 0.2rem;
    }
    .metric-card.teal .value { color: #0D7377; }
    .metric-card.red .value  { color: #D64545; }
    .metric-card.amber .value { color: #E8871E; }
    .metric-card.green .value { color: #2D8B4E; }

    /* ── Result Cards ── */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .result-card .hospital-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1A2332;
    }
    .result-card .hospital-meta {
        font-size: 0.8rem;
        color: #5A6B7F;
        margin-top: 0.2rem;
    }
    .result-card .hospital-desc {
        font-size: 0.85rem;
        color: #3D4F5F;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-high    { background: #D1FAE5; color: #065F46; }
    .badge-medium  { background: #FEF3C7; color: #92400E; }
    .badge-low     { background: #FEE2E2; color: #991B1B; }
    .badge-critical { background: #FEE2E2; color: #991B1B; }
    .badge-high-risk { background: #FFEDD5; color: #9A3412; }
    .badge-moderate { background: #FEF9C3; color: #854D0E; }
    .badge-adequate { background: #D1FAE5; color: #065F46; }

    /* ── Risk indicators ── */
    .risk-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .risk-critical { background-color: #D64545; }
    .risk-high     { background-color: #E8871E; }
    .risk-moderate { background-color: #D4A72C; }
    .risk-adequate { background-color: #2D8B4E; }

    /* ── AI Answer Box ── */
    .ai-answer {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1A2332;
    }
    .ai-answer .answer-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #0D7377;
        margin-bottom: 0.5rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A2332;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0D7377;
        display: inline-block;
    }

    /* ── Gap Analysis Table ── */
    .gap-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }
    .gap-table th {
        background: #F1F5F9;
        color: #475569;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 2px solid #E2E8F0;
    }
    .gap-table td {
        padding: 0.65rem 1rem;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
    }
    .gap-table tr:hover td {
        background: #F8FAFC;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        color: #5A6B7F;
        padding: 0.75rem 1.25rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #0D7377 !important;
        border-bottom-color: #0D7377 !important;
        background: transparent !important;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Anomaly warning ── */
    .anomaly-flag {
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        padding: 0.5rem 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #92400E;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and deduplicate hospital data."""
    for path in ["data/hospital_metadata.csv", "hospital_metadata.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("Hospital data file not found. Place `hospital_metadata.csv` in `data/` folder.")
        st.stop()

    df = df.fillna("")

    # ── Deduplication: keep the row with most data per hospital name ──
    def completeness_score(row):
        score = 0
        for col in ['description', 'procedure', 'equipment', 'capability', 'specialties', 'address_city']:
            val = str(row.get(col, '')).strip()
            if val not in ['', 'nan', '[]', "['']"]:
                score += 1
        return score

    df['_completeness'] = df.apply(completeness_score, axis=1)
    df = df.sort_values('_completeness', ascending=False).drop_duplicates(subset=['name'], keep='first')
    df = df.drop(columns=['_completeness'])
    df = df.reset_index(drop=True)

    # ── Normalize region names to Ghana's 16 official regions ──
    region_fixes = {
        # Greater Accra variants
        'Accra East': 'Greater Accra', 'Accra North': 'Greater Accra',
        'Ga East Municipality': 'Greater Accra',
        'Ga East Municipality, Greater Accra Region': 'Greater Accra',
        'Ledzokuku-Krowor': 'Greater Accra',
        'Kpone Katamanso': 'Greater Accra',
        'Shai Osudoku District, Greater Accra Region': 'Greater Accra',
        'Tema West Municipal': 'Greater Accra',
        # Ashanti variants
        'Asokwa-Kumasi': 'Ashanti', 'Ejisu Municipal': 'Ashanti',
        'Ahafo Ano South-East': 'Ashanti', 'SH': 'Ashanti',
        # Central variants
        'KEEA': 'Central', 'Central Ghana': 'Central',
        # Volta variants
        'Central Tongu District': 'Volta',
        # Ahafo variants
        'Asutifi South': 'Ahafo',
        # Bono / Brong Ahafo
        'Bono': 'Brong Ahafo', 'Dormaa East': 'Brong Ahafo',
        'Techiman Municipal': 'Brong Ahafo',
        # Western variants
        'Takoradi': 'Western',
        # Upper West variants
        'Sissala West District': 'Upper West',
        # Vague / country-level
        'Ghana': 'Unknown',
    }
    df['region_clean'] = df['region_clean'].replace(region_fixes)
    df.loc[df['region_clean'].isin(['', 'nan', 'Unknown']), 'region_clean'] = 'Unknown'

    return df


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

    # Build searchable text per hospital
    def build_text(row):
        parts = []
        for col in ['description', 'procedure', 'equipment', 'capability', 'specialties', 'search_text_full']:
            val = str(row.get(col, ''))
            if val not in ['', 'nan', '[]', "['']"]:
                parts.append(val)
        return ' '.join(parts).lower()

    df_copy = df.copy()
    df_copy['_all_text'] = df_copy.apply(build_text, axis=1)

    # Flag services per hospital
    for svc_name, keywords in services.items():
        df_copy[f'has_{svc_name}'] = df_copy['_all_text'].apply(
            lambda t: any(k in t for k in keywords)
        )

    # Aggregate by region
    svc_cols = [f'has_{s}' for s in services]
    region_scores = df_copy[df_copy['region_clean'] != 'Unknown'].groupby('region_clean').agg(
        total_facilities=('name', 'count'),
        **{col: (col, 'sum') for col in svc_cols}
    ).reset_index()

    # Services available count (binary: does the region have at least one?)
    region_scores['services_available'] = sum(
        (region_scores[f'has_{s}'] > 0).astype(int) for s in services
    )

    # Risk level
    def get_risk(row):
        s = row['services_available']
        f = row['total_facilities']
        if s <= 2 or f <= 3:
            return 'Critical'
        elif s <= 4 or f <= 8:
            return 'High Risk'
        elif s <= 6 or f <= 20:
            return 'Moderate'
        return 'Adequate'

    region_scores['risk_level'] = region_scores.apply(get_risk, axis=1)
    region_scores = region_scores.sort_values('services_available')

    return region_scores, services


# ─── FAISS + RAG ─────────────────────────────────────────────
@st.cache_resource
def load_faiss(df):
    """Build FAISS index for semantic search."""
    from sentence_transformers import SentenceTransformer
    import faiss

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def build_text(row):
        parts = [f"Hospital: {row['name']}", f"Region: {row['region_clean']}"]
        for col in ['facilityTypeId', 'address_city', 'description',
                     'specialties', 'procedure', 'capability', 'equipment']:
            val = str(row.get(col, ''))
            if val not in ['', 'nan', '[]', "['']"]:
                parts.append(f"{col}: {val[:300]}")
        return ' | '.join(parts)

    texts = df.apply(build_text, axis=1).tolist()
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=64)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    return embedder, index, embeddings


def search_hospitals(query, df, embedder, index, top_k=5, region_filter=None):
    """Semantic search with optional region filtering."""
    import faiss as faiss_lib

    if region_filter and region_filter != "All":
        region_mask = df['region_clean'] == region_filter
        region_indices = df[region_mask].index.tolist()
        if len(region_indices) >= 3:
            region_embeddings = np.array([
                embedder.encode([df.iloc[i].to_dict().get('name', '')]) for i in region_indices
            ]).squeeze()
            # Rebuild for region subset (use cached full embeddings approach)
            pass

    q_emb = embedder.encode([query]).astype('float32')
    distances, indices = index.search(q_emb, min(top_k, len(df)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        similarity = round(1 / (1 + dist), 3)
        results.append({
            'name': row['name'],
            'region': row['region_clean'],
            'city': row.get('address_city', ''),
            'type': row.get('facilityTypeId', ''),
            'description': str(row.get('description', ''))[:300],
            'procedure': str(row.get('procedure', '')),
            'capability': str(row.get('capability', '')),
            'equipment': str(row.get('equipment', '')),
            'specialties': str(row.get('specialties', '')),
            'confidence': row.get('confidence', 'Medium'),
            'similarity': similarity,
        })
    return results


def detect_anomalies(result):
    """Check for suspicious claims in a hospital result."""
    anomalies = []
    ftype = str(result.get('type', '')).lower()
    all_text = (result.get('procedure', '') + ' ' +
                result.get('capability', '') + ' ' +
                result.get('equipment', '')).lower()

    if ftype == 'clinic' and 'icu' in all_text:
        anomalies.append('Clinic claiming ICU capability — verify')
    if ftype == 'pharmacy' and 'surgery' in all_text:
        anomalies.append('Pharmacy claiming surgery — suspicious')
    if 'icu' in all_text and 'emergency' not in all_text:
        anomalies.append('Claims ICU but no emergency services listed')
    return anomalies


def get_llm_answer(question, results, groq_key):
    """Get AI answer using retrieved hospital context."""
    if not groq_key:
        return "Set the GROQ_KEY environment variable to enable AI answers."

    from groq import Groq
    client = Groq(api_key=groq_key)

    context = ""
    for i, r in enumerate(results, 1):
        context += f"""
Hospital {i}: {r['name']}
  Region: {r['region']}
  Type: {r['type']}
  Confidence: {r['confidence']}
  Procedures: {r['procedure'][:200]}
  Capabilities: {r['capability'][:200]}
  Equipment: {r['equipment'][:200]}
  Specialties: {r['specialties'][:200]}
---"""

    prompt = f"""You are a healthcare analyst for Virtue Foundation in Ghana.
You help NGO coordinators find hospitals and plan deployments.

Hospital data from our database:
{context}

Question: {question}

Instructions:
- Be specific about which hospitals can help and their region
- If data confidence is Low, mention it needs verification
- If a region has few hospitals, note it as a potential service gap
- If none match well, say so and suggest what's available
- Keep answer under 200 words
- Be factual, not promotional"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting AI response: {str(e)[:100]}"


# ─── Load Data ───────────────────────────────────────────────
df = load_data()
gap_df, services_dict = build_gap_analysis(df)
groq_key = os.environ.get("GROQ_KEY", "")

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    regions = sorted(df[df['region_clean'] != 'Unknown']['region_clean'].unique().tolist())
    selected_region = st.selectbox("Region", ["All"] + regions, index=0)

    facility_types = sorted(df[df['facilityTypeId'] != '']['facilityTypeId'].unique().tolist())
    selected_type = st.selectbox("Facility type", ["All"] + facility_types, index=0)

    risk_levels = ['All', 'Critical', 'High Risk', 'Moderate', 'Adequate']
    selected_risk = st.selectbox("Risk level", risk_levels, index=0)

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**{len(df)}** unique facilities")
    st.markdown(f"**{df['region_clean'].nunique() - (1 if 'Unknown' in df['region_clean'].values else 0)}** regions covered")

    known_regions = df[df['region_clean'] != 'Unknown']
    critical_count = len(gap_df[gap_df['risk_level'] == 'Critical'])
    st.markdown(f"**{critical_count}** critical desert regions")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#8896A6;line-height:1.4'>"
        "Virtue Foundation · Ghana<br>"
        "Healthcare coverage analysis tool<br>"
        "Data: Virtue Foundation Ghana v0.3"
        "</div>",
        unsafe_allow_html=True
    )

# ─── Apply Filters ───────────────────────────────────────────
filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['region_clean'] == selected_region]
if selected_type != "All":
    filtered_df = filtered_df[filtered_df['facilityTypeId'] == selected_type]


# ─── Header ──────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<h1>Ghana Healthcare Coverage</h1>'
    '<p>Identify medical deserts and plan NGO resource deployment across Ghana\'s 16 regions</p>'
    '</div>',
    unsafe_allow_html=True
)


# ─── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Search", "Regional Analysis", "Map", "Hospital Directory"
])


# ══════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card teal">'
            f'<div class="label">Total Facilities</div>'
            f'<div class="value">{len(filtered_df)}</div>'
            f'<div class="subtext">{"All regions" if selected_region == "All" else selected_region}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card red">'
            f'<div class="label">Critical Deserts</div>'
            f'<div class="value">{critical_count}</div>'
            f'<div class="subtext">Regions needing urgent support</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col3:
        svc_count = len(services_dict)
        st.markdown(
            f'<div class="metric-card amber">'
            f'<div class="label">Services Tracked</div>'
            f'<div class="value">{svc_count}</div>'
            f'<div class="subtext">ICU, ER, Surgery, Lab, etc.</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col4:
        regions_covered = df[df['region_clean'] != 'Unknown']['region_clean'].nunique()
        st.markdown(
            f'<div class="metric-card green">'
            f'<div class="label">Regions Covered</div>'
            f'<div class="value">{regions_covered}</div>'
            f'<div class="subtext">Out of 16 Ghana regions</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("")

    # Charts
    import plotly.express as px
    import plotly.graph_objects as go

    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown('<div class="section-header">Facilities by Region</div>', unsafe_allow_html=True)
        region_counts = (
            filtered_df[filtered_df['region_clean'] != 'Unknown']
            .groupby('region_clean')['name']
            .count()
            .reset_index()
        )
        if not region_counts.empty:
            region_counts.columns = ['Region', 'Count']
            region_counts = region_counts.sort_values('Count', ascending=True)

            # Merge risk levels
            risk_map = dict(zip(gap_df['region_clean'], gap_df['risk_level']))
            region_counts['Risk'] = region_counts['Region'].map(risk_map).fillna('Unknown')

            color_scale = {
                'Critical': '#D64545', 'High Risk': '#E8871E',
                'Moderate': '#D4A72C', 'Adequate': '#2D8B4E', 'Unknown': '#94A3B8'
            }
            region_counts['Color'] = region_counts['Risk'].map(color_scale)

            fig_bar = go.Figure(go.Bar(
                x=region_counts['Count'],
                y=region_counts['Region'],
                orientation='h',
                marker_color=region_counts['Color'],
                text=region_counts['Count'],
                textposition='outside',
                textfont=dict(size=11, color='#475569'),
            ))
            fig_bar.update_layout(
                height=max(360, len(region_counts) * 28),
                margin=dict(l=0, r=40, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=''),
                yaxis=dict(title=''),
                font=dict(family='Inter', size=12),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = gap_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']

        color_order = {'Critical': '#D64545', 'High Risk': '#E8871E',
                       'Moderate': '#D4A72C', 'Adequate': '#2D8B4E'}

        fig_donut = go.Figure(go.Pie(
            labels=risk_counts['Risk Level'],
            values=risk_counts['Count'],
            hole=0.55,
            marker_colors=[color_order.get(r, '#94A3B8') for r in risk_counts['Risk Level']],
            textinfo='label+value',
            textfont=dict(size=12, family='Inter'),
            hovertemplate='%{label}: %{value} regions<extra></extra>',
        ))
        fig_donut.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family='Inter'),
            annotations=[dict(
                text=f"<b>{len(gap_df)}</b><br>Regions",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font=dict(family='Inter', color='#475569')
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Service heatmap
    st.markdown('<div class="section-header">Service Coverage Heatmap</div>', unsafe_allow_html=True)
    svc_cols = [f'has_{s}' for s in services_dict]
    heatmap_data = gap_df.set_index('region_clean')[svc_cols].copy()
    heatmap_data.columns = list(services_dict.keys())
    heatmap_data = heatmap_data.map(lambda x: 1 if x > 0 else 0)
    heatmap_data = heatmap_data.sort_index()

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, '#FEE2E2'], [1, '#D1FAE5']],
        showscale=False,
        text=heatmap_data.values,
        texttemplate='%{text}',
        textfont=dict(size=11, family='Inter'),
        hovertemplate='%{y} · %{x}: %{z}<extra></extra>',
        xgap=2, ygap=2,
    ))
    fig_heat.update_layout(
        height=max(300, len(heatmap_data) * 28),
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        xaxis=dict(side='top'),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("1 = at least one facility offers this service in the region · 0 = no facility offers it")


# ══════════════════════════════════════════════════════════════
# TAB 2: SEARCH
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Search Healthcare Facilities</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#5A6B7F;font-size:0.85rem;margin-top:-0.5rem'>"
        "Ask questions about hospital services, regional coverage, or specific capabilities.</p>",
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Search",
        placeholder="e.g. Which hospitals in Northern Ghana have emergency care?",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Searching..."):
            embedder, index, embeddings = load_faiss(df)
            results = search_hospitals(query, df, embedder, index, top_k=5)

        # AI Answer
        if groq_key:
            with st.spinner("Generating analysis..."):
                answer = get_llm_answer(query, results, groq_key)
            st.markdown(
                f'<div class="ai-answer">'
                f'<div class="answer-label">Analysis</div>'
                f'{answer}'
                f'</div>',
                unsafe_allow_html=True
            )
        elif not groq_key:
            st.info("Set the `GROQ_KEY` environment variable to enable AI-powered analysis.")

        # Results
        st.markdown(f'<div class="section-header">Matching Facilities ({len(results)})</div>',
                     unsafe_allow_html=True)

        for r in results:
            conf = r.get('confidence', 'Medium')
            badge_class = 'badge-high' if conf == 'High' else 'badge-medium' if conf == 'Medium' else 'badge-low'

            anomalies = detect_anomalies(r)
            anomaly_html = ""
            for a in anomalies:
                anomaly_html += f'<div class="anomaly-flag">{a}</div>'

            desc_text = r['description'][:250] + "..." if len(r['description']) > 250 else r['description']
            desc_html = f'<div class="hospital-desc">{desc_text}</div>' if desc_text else ""

            st.markdown(
                f'<div class="result-card">'
                f'<div class="hospital-name">{r["name"]}'
                f' <span class="badge {badge_class}">{conf} confidence</span></div>'
                f'<div class="hospital-meta">'
                f'{r["region"]} · {r["city"]} · {r["type"]} · Match: {r["similarity"]}'
                f'</div>'
                f'{desc_html}'
                f'{anomaly_html}'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
# TAB 3: REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Regional Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#5A6B7F;font-size:0.85rem;margin-top:-0.5rem'>"
        "Service coverage and risk assessment across Ghana's regions.</p>",
        unsafe_allow_html=True
    )

    # Filter gap data by selected risk
    display_gap = gap_df.copy()
    if selected_risk != "All":
        display_gap = display_gap[display_gap['risk_level'] == selected_risk]

    # Build HTML table
    table_html = '<table class="gap-table"><thead><tr>'
    table_html += '<th>Region</th><th>Facilities</th><th>Services</th><th>Risk</th>'
    for svc in services_dict:
        table_html += f'<th>{svc}</th>'
    table_html += '</tr></thead><tbody>'

    for _, row in display_gap.iterrows():
        risk = row['risk_level']
        risk_class = risk.lower().replace(' ', '-')
        dot_class = f'risk-{risk_class}' if risk_class != 'high-risk' else 'risk-high'

        table_html += '<tr>'
        table_html += f'<td><strong>{row["region_clean"]}</strong></td>'
        table_html += f'<td>{int(row["total_facilities"])}</td>'
        table_html += f'<td>{int(row["services_available"])}/8</td>'
        table_html += (
            f'<td><span class="risk-dot {dot_class}"></span>'
            f'<span class="badge badge-{risk_class}">{risk}</span></td>'
        )

        for svc in services_dict:
            has_it = row.get(f'has_{svc}', 0)
            icon = "✓" if has_it > 0 else "✗"
            color = "#2D8B4E" if has_it > 0 else "#D64545"
            table_html += f'<td style="color:{color};font-weight:600;text-align:center">{icon}</td>'

        table_html += '</tr>'

    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)

    # Deployment recommendations
    critical_regions = gap_df[gap_df['risk_level'] == 'Critical']
    if not critical_regions.empty:
        st.markdown("")
        st.markdown('<div class="section-header">Deployment Recommendations</div>', unsafe_allow_html=True)

        for _, row in critical_regions.iterrows():
            missing = []
            for svc in services_dict:
                if row.get(f'has_{svc}', 0) == 0:
                    missing.append(svc)

            st.markdown(
                f'<div class="result-card">'
                f'<div class="hospital-name">'
                f'<span class="risk-dot risk-critical"></span>{row["region_clean"]}'
                f' <span class="badge badge-critical">Critical</span></div>'
                f'<div class="hospital-meta">'
                f'{int(row["total_facilities"])} facilities · '
                f'{int(row["services_available"])}/8 services available</div>'
                f'<div class="hospital-desc" style="margin-top:0.5rem">'
                f'<strong>Missing services:</strong> {", ".join(missing)}<br>'
                f'<strong>Recommended:</strong> Deploy medical team with '
                f'{missing[0] if missing else "general"} capability as first priority.</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
# TAB 4: MAP
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Medical Desert Map</div>', unsafe_allow_html=True)

    map_path = None
    for path in ["data/ghana_map.html", "ghana_map.html"]:
        if os.path.exists(path):
            map_path = path
            break

    if map_path:
        with open(map_path, 'r', encoding='utf-8') as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=600, scrolling=False)
        st.caption("Circle size represents number of facilities. Click a region for details.")
    else:
        st.warning("Map file not found. Place `ghana_map.html` in the `data/` folder.")


# ══════════════════════════════════════════════════════════════
# TAB 5: HOSPITAL DIRECTORY
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Hospital Directory</div>', unsafe_allow_html=True)

    display_cols = ['name', 'region_clean', 'address_city', 'facilityTypeId']
    optional_cols = ['specialties', 'confidence']
    for c in optional_cols:
        if c in filtered_df.columns:
            display_cols.append(c)

    st.dataframe(
        filtered_df[display_cols].rename(columns={
            'name': 'Hospital',
            'region_clean': 'Region',
            'address_city': 'City',
            'facilityTypeId': 'Type',
            'specialties': 'Specialties',
            'confidence': 'Confidence',
        }),
        use_container_width=True,
        height=500,
    )

    # SQL query (collapsed)
    with st.expander("Advanced: Custom SQL Query"):
        st.markdown(
            "<p style='font-size:0.8rem;color:#5A6B7F'>"
            "Run SQL queries against the hospital dataset. Table name: <code>hospitals</code></p>",
            unsafe_allow_html=True
        )
        sql_query = st.text_area(
            "SQL",
            value="SELECT name, region_clean, facilityTypeId FROM hospitals WHERE region_clean = 'Greater Accra' LIMIT 10",
            height=80,
            label_visibility="collapsed"
        )
        if st.button("Run Query"):
            try:
                from pandasql import sqldf
                hospitals = df
                result = sqldf(sql_query, {'hospitals': hospitals})
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {str(e)}")