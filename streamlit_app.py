import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ast
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ════════════════════════════════════════════════════════════════
def unpack_list(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None', '[""]']:
        return ""
    s = str(val).strip()
    try:    parsed = json.loads(s)
    except:
        try:    parsed = ast.literal_eval(s)
        except: return s
    if isinstance(parsed, list):
        items = [str(x).strip() for x in parsed if str(x).strip() not in ['', 'nan', '""', "''"]]
        return " | ".join(items)
    return str(val)


def clean_capability(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return ""
    # Handle pipe-separated enriched format (from enriched_capability)
    if " | " in str(val) and not str(val).startswith("["):
        items = [i.strip() for i in str(val).split(" | ") if i.strip()]
        clean = []
        for item in items:
            s = item.lower()
            if any(x in s for x in ['@','www.','http','+233','located at',
                                     'contact','opening hours','closed on','facebook',
                                     'instagram','twitter']):
                continue
            clean.append(item)
        return " | ".join(clean)
    # Handle JSON list format
    try:    parsed = json.loads(str(val))
    except:
        try:    parsed = ast.literal_eval(str(val))
        except: return str(val)
    if isinstance(parsed, list):
        clean = []
        for item in parsed:
            s = str(item).lower()
            if any(x in s for x in ['contact phone','email','@','www.','located at',
                                     'currently closed','opening hours','closed on','http',
                                     'facebook','instagram','twitter','+233']):
                continue
            clean.append(str(item).strip())
        return " | ".join(clean)
    return str(val)


def build_search_text(row):
    parts = [
        f"Hospital: {row.get('name','')}",
        f"Region: {row.get('region_clean','')}",
        f"Type: {row.get('facilityTypeId','')}",
        f"City: {row.get('address_city','')}",
    ]
    desc = str(row.get('description', '')).strip()
    if desc and desc not in ('nan', ''):
        parts.append(f"Description: {desc[:300]}")
    for label, col in [('Specialties','specialties_text'),('Procedures','procedure_text'),
                       ('Capabilities','capability_text'),('Equipment','equipment_text')]:
        v = str(row.get(col, '')).strip()
        if v and v != 'nan':
            parts.append(f"{label}: {v[:400]}")
    return " | ".join(parts)


# ════════════════════════════════════════════════════════════════
# REGION / SERVICE DETECTION
# ════════════════════════════════════════════════════════════════
REGION_MAP = {
    'Greater Accra': ['accra','greater accra','tema','legon','madina','spintex','cantonments'],
    'Ashanti':       ['ashanti','kumasi','obuasi','ejisu'],
    'Northern':      ['northern','tamale','northern ghana','yendi','savelugu'],
    'Upper East':    ['upper east','bolgatanga','bawku','navrongo'],
    'Upper West':    ['upper west','wa ','lawra','jirapa'],
    'Volta':         ['volta','hohoe','aflao','keta','kpando'],
    'Western':       ['western','takoradi','sekondi','tarkwa'],
    'Central':       ['central','cape coast','winneba','elmina'],
    'Eastern':       ['eastern','koforidua','nkawkaw','akosombo'],
    'Brong Ahafo':   ['brong','sunyani','techiman','berekum'],
    'Savannah':      ['savannah','damongo','bole'],
    'North East':    ['north east','nalerigu','gambaga'],
    'Oti':           ['oti','dambai','nkwanta'],
    'Ahafo':         ['ahafo','goaso','bechem'],
    'Western North': ['western north','bibiani','sefwi'],
    'Bono East':     ['bono east','atebubu'],
}

SERVICE_MAP = {
    'icu':        ['icu','intensive care','critical care','nicu'],
    'emergency':  ['emergency','accident','24/7','24 hour','24-hour','a&e'],
    'surgery':    ['surgery','surgical','operating theatre','operation','theatr'],
    'maternity':  ['maternity','obstetric','gynecol','gynaecol','delivery','labour','antenatal','prenatal'],
    'laboratory': ['laboratory','lab test','diagnostic lab','pathology','blood test'],
    'imaging':    ['mri','ct scan','x-ray','xray','imaging','radiology','ultrasound','mammogram'],
    'pediatrics': ['pediatric','paediatric','children hospital','neonatal','child care'],
    'pharmacy':   ['pharmacy','pharmacist','dispensary','pharmaceutical'],
    'dental':     ['dental','dentist','dentistry','orthodon'],
    'cardiac':    ['cardiac','cardiology','heart surgery','echocardiography'],
}

QUERY_SYNONYMS = {
    "icu":        ["icu","intensive care unit","critical care","intensive care"],
    "emergency":  ["emergency","accident and emergency","trauma","24 hour","urgent care"],
    "surgery":    ["surgery","surgical","operating theatre","operation"],
    "maternity":  ["maternity","obstetric","delivery","labour","antenatal"],
    "pediatric":  ["pediatric","children","neonatal","child health"],
    "imaging":    ["imaging","x-ray","xray","mri","ct scan","ultrasound","radiology"],
    "laboratory": ["laboratory","lab","diagnostic testing","pathology"],
    "pharmacy":   ["pharmacy","pharmaceutical","dispensary"],
    "dental":     ["dental","dentistry","tooth"],
}

def detect_region(query):
    q = query.lower()
    for region, keywords in REGION_MAP.items():
        if any(kw in q for kw in keywords):
            return region
    return None

def detect_service(query):
    q = query.lower()
    for service, keywords in SERVICE_MAP.items():
        if any(kw in q for kw in keywords):
            return service, keywords
    return None, []

def expand_query(query):
    q = query.lower()
    expansions = []
    for key, synonyms in QUERY_SYNONYMS.items():
        if key in q or any(s in q for s in synonyms[:2]):
            expansions.extend(synonyms)
    return (q + " " + " ".join(expansions)).strip() if expansions else q

def is_gap_analysis_question(query):
    q = query.lower()
    return any(x in q for x in ['desert','deploy','urgent','shortage','fewest','least hospital',
                                  'which region','how many region','coverage','underserved'])


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# FIX: Use enriched_capability if present in CSV
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = None
    for path in ["data/hospital_metadata.csv", "hospital_metadata.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    if df is None:
        st.error("hospital_metadata.csv not found. Place it in the data/ folder.")
        st.stop()

    df = df.fillna("")

    # FIX: Use enriched_capability column if it exists and is populated
    # This is exported from Databricks pipeline with specialties + description inference
    has_enriched = (
        'enriched_capability' in df.columns and
        df['enriched_capability'].str.len().mean() > 10
    )

    df['procedure_text']   = df['procedure'].apply(unpack_list)
    df['equipment_text']   = df['equipment'].apply(unpack_list)
    df['specialties_text'] = df['specialties'].apply(unpack_list)

    if has_enriched:
        # Use enriched capability — much richer (ICU: 35+, Surgery: 200+)
        df['capability_text'] = df['enriched_capability'].apply(clean_capability)
    else:
        df['capability_text'] = df['capability'].apply(clean_capability)

    # FIX: build_search_text now uses enriched capability_text
    df['search_text_full'] = df.apply(build_search_text, axis=1)

    # Dedup — keep most complete row per hospital name
    def completeness(row):
        return sum(1 for c in ['procedure_text','equipment_text','capability_text',
                               'specialties_text','address_city']
                   if str(row.get(c,'')).strip() not in ['','nan'])
    df['_c'] = df.apply(completeness, axis=1)
    df = df.sort_values('_c', ascending=False).drop_duplicates(subset=['name'], keep='first')
    df = df.drop(columns=['_c']).reset_index(drop=True)

    # Region normalization
    region_fixes = {
        'Accra East':'Greater Accra','Accra North':'Greater Accra',
        'Ga East Municipality':'Greater Accra',
        'Ga East Municipality, Greater Accra Region':'Greater Accra',
        'Ledzokuku-Krowor':'Greater Accra','Kpone Katamanso':'Greater Accra',
        'Shai Osudoku District, Greater Accra Region':'Greater Accra',
        'Tema West Municipal':'Greater Accra',
        'Asokwa-Kumasi':'Ashanti','Ejisu Municipal':'Ashanti',
        'Ahafo Ano South-East':'Ashanti','SH':'Ashanti',
        'KEEA':'Central','Central Ghana':'Central','Central Tongu District':'Volta',
        'Asutifi South':'Ahafo','Bono':'Brong Ahafo','Dormaa East':'Brong Ahafo',
        'Techiman Municipal':'Brong Ahafo','Takoradi':'Western',
        'Sissala West District':'Upper West','Ghana':'Unknown',
    }
    df['region_clean'] = df['region_clean'].replace(region_fixes)
    df.loc[df['region_clean'].isin(['','nan','Unknown']),'region_clean'] = 'Unknown'

    city_to_region = {
        'accra':'Greater Accra','tema':'Greater Accra','madina':'Greater Accra',
        'legon':'Greater Accra','adenta':'Greater Accra','dansoman':'Greater Accra',
        'lapaz':'Greater Accra','achimota':'Greater Accra','nungua':'Greater Accra',
        'weija':'Greater Accra','ashaiman':'Greater Accra','amasaman':'Greater Accra',
        'cantonments':'Greater Accra','east legon':'Greater Accra',
        'kumasi':'Ashanti','obuasi':'Ashanti','ejisu':'Ashanti',
        'takoradi':'Western','sekondi':'Western','tarkwa':'Western',
        'cape coast':'Central','winneba':'Central','saltpond':'Central',
        'tamale':'Northern','yendi':'Northern','savelugu':'Northern',
        'bolgatanga':'Upper East','bawku':'Upper East','navrongo':'Upper East',
        'wa':'Upper West','lawra':'Upper West',
        'hohoe':'Volta','kpando':'Volta','aflao':'Volta',
        'sunyani':'Brong Ahafo','techiman':'Brong Ahafo',
        'akosombo':'Eastern','koforidua':'Eastern','asamankese':'Eastern',
        'sefwi':'Western North','bibiani':'Western North',
        'damongo':'Savannah','nalerigu':'North East','gambaga':'North East',
        'dambai':'Oti','nkwanta':'Oti',
    }
    def fix_unknown(row):
        if row['region_clean'] != 'Unknown': return row['region_clean']
        combined = (str(row.get('address_city','')).lower() + ' ' + str(row.get('name','')).lower())
        for k, v in city_to_region.items():
            if k in combined: return v
        return 'Unknown'
    df['region_clean'] = df.apply(fix_unknown, axis=1)
    return df


# ════════════════════════════════════════════════════════════════
# HYBRID SEARCH INDEX
# FIX: cache_resource can't hash DataFrames — use row count + col
# count as a cache key instead, pass df separately
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_search_index(_df_len, _df_cols):
    """
    FIX: @st.cache_resource cannot hash DataFrames.
    We pass df length + columns as hashable proxy.
    The actual df is accessed via st.session_state inside the function.
    """
    df = st.session_state.get("_search_df")
    if df is None:
        return None, None, None, None, []

    def build_rich_text(row):
        parts = [
            f"Hospital: {row.get('name','')}",
            f"Region: {row.get('region_clean','')}",
            f"City: {row.get('address_city','')}",
            f"Type: {row.get('facilityTypeId','')}",
        ]
        desc = str(row.get('description','')).strip()
        if desc and desc != 'nan':
            parts.append(f"Description: {desc[:300]}")

        specs = str(row.get('specialties_text','')).strip()
        if specs and specs != 'nan':
            parts.append(f"Specialties: {specs}")
            parts.append(f"Specialties: {specs}")  # boost

        cap = str(row.get('capability_text','')).strip()
        if cap and cap != 'nan':
            # Boost 3x — most important signal
            parts.append(f"Capabilities: {cap}")
            parts.append(f"Capabilities: {cap}")
            parts.append(f"Capabilities: {cap}")

        return " | ".join(parts)

    texts = df.apply(build_rich_text, axis=1).tolist()

    # BM25
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # FAISS — load pre-built file if available, else build fresh
    import faiss as faiss_lib
    from sentence_transformers import SentenceTransformer

    embedder    = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_path  = next((p for p in ["data/hospital_index.faiss","hospital_index.faiss"] if os.path.exists(p)), None)
    emb_path    = next((p for p in ["data/hospital_embeddings.npy","hospital_embeddings.npy"] if os.path.exists(p)), None)

    if faiss_path and emb_path:
        faiss_index = faiss_lib.read_index(faiss_path)
        embeddings  = np.load(emb_path)
    else:
        embeddings  = embedder.encode(texts, show_progress_bar=False, batch_size=64).astype('float32')
        faiss_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)

    return bm25, faiss_index, embeddings, embedder, texts


def hybrid_search(query, df, top_k=8, region_override=None):
    """
    Hybrid BM25 (40%) + FAISS (60%) with region filter + query expansion.
    """
    region  = region_override or detect_region(query)
    service, svc_keywords = detect_service(query)

    # Store df in session state so cache_resource function can access it
    st.session_state["_search_df"] = df

    index_result = load_search_index(len(df), str(df.columns.tolist()))
    bm25, faiss_index, embeddings, embedder, texts = index_result

    if bm25 is None:
        # Fallback to keyword-only if index failed
        return _keyword_fallback(query, df, region, service, svc_keywords, top_k)

    query_expanded = expand_query(query)

    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(query_expanded.lower().split()))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # FAISS scores
    import faiss as faiss_lib
    q_emb = embedder.encode([query_expanded]).astype('float32')
    dists, idxs = faiss_index.search(q_emb, len(df))
    faiss_scores = np.zeros(len(df))
    for dist, idx in zip(dists[0], idxs[0]):
        if idx < len(df):
            faiss_scores[idx] = 1 / (1 + dist)
    if faiss_scores.max() > 0:
        faiss_scores = faiss_scores / faiss_scores.max()

    hybrid_scores = 0.4 * bm25_scores + 0.6 * faiss_scores

    # Build results
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        results.append({
            'name':        row['name'],
            'region':      row['region_clean'],
            'city':        str(row.get('address_city', '')),
            'type':        str(row.get('facilityTypeId', '')),
            'capability':  str(row.get('capability_text', '')),
            'procedure':   str(row.get('procedure_text', '')),
            'equipment':   str(row.get('equipment_text', '')),
            'specialties': str(row.get('specialties_text', '')),
            'confidence':  str(row.get('confidence', 'Medium')),
            'score':       float(hybrid_scores[i]),
        })

    # Hard region filter
    if region:
        filtered = [r for r in results if region.lower() in r['region'].lower()]
        if filtered:
            results = filtered

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Mark service matches
    for r in results:
        all_text = (r['capability'] + ' ' + r['procedure'] + ' ' + r['specialties']).lower()
        r['score_display'] = int(
            bool(svc_keywords) and any(k in all_text for k in svc_keywords)
        )

    return results[:top_k], region, service


def _keyword_fallback(query, df, region, service, svc_keywords, top_k):
    """Simple keyword fallback if index not ready."""
    pool = df.copy()
    if region:
        pool = df[df['region_clean'] == region].copy()
        if len(pool) == 0:
            pool = df.copy()

    if svc_keywords:
        def score_row(text):
            t = text.lower()
            return sum(1 for kw in svc_keywords if kw in t)
        pool['_score'] = pool['search_text_full'].apply(score_row)
        pool = pool.sort_values('_score', ascending=False)
    else:
        pool['_score'] = 0

    results = []
    for _, row in pool.head(top_k).iterrows():
        results.append({
            'name':        row['name'],
            'region':      row['region_clean'],
            'city':        str(row.get('address_city', '')),
            'type':        str(row.get('facilityTypeId', '')),
            'capability':  str(row.get('capability_text', '')),
            'procedure':   str(row.get('procedure_text', '')),
            'equipment':   str(row.get('equipment_text', '')),
            'specialties': str(row.get('specialties_text', '')),
            'confidence':  str(row.get('confidence', 'Medium')),
            'score':       float(row.get('_score', 0)),
            'score_display': int(row.get('_score', 0) > 0),
        })
    return results, region, service


# ════════════════════════════════════════════════════════════════
# GAP ANALYSIS
# FIX: Use search_text_full which is built from enriched capability_text
# ════════════════════════════════════════════════════════════════
@st.cache_data
def build_gap_analysis(df):
    services = {
        'ICU':        ['icu','intensive care','critical care'],
        'Emergency':  ['emergency','accident','24/7','24 hour'],
        'Surgery':    ['surgery','surgical','operating theatre'],
        'Maternity':  ['maternity','obstetric','gynecology','delivery','gynaecology'],
        'Laboratory': ['laboratory','lab test','diagnostic lab'],
        'Imaging':    ['x-ray','xray','ultrasound','mri','ct scan','radiology'],
        'Pediatrics': ['pediatric','paediatric','children','neonatal'],
        'Pharmacy':   ['pharmacy','pharmaceutical','dispensary'],
    }
    # FIX: Use search_text_full (which includes enriched capability_text)
    df2 = df.copy()
    for svc, kws in services.items():
        df2[f'has_{svc}'] = df2['search_text_full'].str.lower().apply(
            lambda t: any(k in t for k in kws)
        )

    svc_cols = [f'has_{s}' for s in services]
    rs = (df2[df2['region_clean'] != 'Unknown']
          .groupby('region_clean')
          .agg(total_facilities=('name','count'), **{c:(c,'sum') for c in svc_cols})
          .reset_index())
    rs['services_available'] = sum((rs[f'has_{s}'] > 0).astype(int) for s in services)

    def risk(row):
        s, f = row['services_available'], row['total_facilities']
        if s <= 2 or f <= 3:    return 'Critical'
        elif s <= 4 or f <= 8:  return 'High Risk'
        elif s <= 6 or f <= 20: return 'Moderate'
        return 'Adequate'
    rs['risk_level'] = rs.apply(risk, axis=1)
    return rs.sort_values('services_available'), services


def get_gap_analysis_answer(df):
    """Structured gap analysis answer using enriched search_text_full."""
    region_counts = df[df['region_clean'] != 'Unknown']['region_clean'].value_counts().sort_values()
    lines = ["**Medical desert regions in Ghana (fewest hospitals):**\n"]
    for region, count in region_counts.items():
        risk = ('🔴 CRITICAL' if count <= 3 else
                '🟠 HIGH RISK' if count <= 8 else
                '🟡 MODERATE' if count <= 20 else '🟢 ADEQUATE')
        rd = df[df['region_clean'] == region]
        # FIX: use search_text_full (includes enriched capability)
        stf = rd['search_text_full'].str.lower()
        has_icu   = stf.str.contains('icu|intensive care', na=False).any()
        has_emerg = stf.str.contains('emergency|24/7', na=False).any()
        has_surg  = stf.str.contains('surg', na=False).any()
        gaps = []
        if not has_icu:   gaps.append('no ICU')
        if not has_emerg: gaps.append('no emergency')
        if not has_surg:  gaps.append('no surgery')
        gap_str = f" — missing: {', '.join(gaps)}" if gaps else ""
        lines.append(f"- **{region}**: {count} hospitals {risk}{gap_str}")
        if count > 20:
            break
    return "\n".join(lines)


def detect_anomalies(result):
    anomalies = []
    ftype    = str(result.get('type', '')).lower()
    all_text = (result.get('procedure','') + ' ' + result.get('capability','') + ' ' + result.get('equipment','')).lower()
    if ftype == 'clinic' and 'icu' in all_text:
        anomalies.append('Clinic claiming ICU — verify before routing patients')
    if ftype == 'pharmacy' and 'surgery' in all_text:
        anomalies.append('Pharmacy claiming surgery — suspicious')
    return anomalies


def get_llm_answer(question, results, groq_key, df, region, service):
    if not groq_key:
        return "Set GROQ_KEY in Streamlit Secrets to enable AI analysis."
    from groq import Groq
    gclient = Groq(api_key=groq_key)

    region_counts = df['region_clean'].value_counts()
    stats = f"Total hospitals in database: {len(df)}\n"
    if region:
        cnt = region_counts.get(region, 0)
        svc_matches = sum(1 for r in results if r.get('score_display', 0) > 0)
        stats += f"Hospitals in {region}: {cnt}\n"
        stats += f"Of these, {svc_matches} have '{service or 'requested'}' services\n"

    context = ""
    for i, r in enumerate(results, 1):
        context += (
            f"Hospital {i}: {r['name']} | Region: {r['region']} | City: {r['city']}\n"
            f"  Capabilities: {r['capability'][:200]}\n"
            f"  Procedures:   {r['procedure'][:150]}\n\n"
        )

    prompt = f"""You are a healthcare analyst for Virtue Foundation in Ghana.

DATABASE STATS:
{stats}

RETRIEVED HOSPITALS (filtered to {region or 'all regions'}{f', sorted by {service} relevance' if service else ''}):
{context}

QUESTION: {question}

Answer rules:
- Name specific hospitals from the list above.
- If asked about ICU: look for 'ICU', 'intensive care' in capabilities.
- If asked about a specific region: ONLY mention hospitals from that region.
- If 0 relevant hospitals found for the service, say so honestly.
- Keep answer under 180 words, factual."""

    try:
        resp = gclient.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            n=1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        if "429" in str(e):
            try:
                resp = gclient.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    n=1,
                )
                return "*(fallback model)*\n\n" + resp.choices[0].message.content.strip()
            except:
                return "⚠️ Rate limited. Wait 1 minute and retry."
        return f"Error: {str(e)[:100]}"


# ════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ghana Healthcare Coverage · Virtue Foundation",
    page_icon="⚕️", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #F7F8FA; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .app-header { padding: 1.5rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.5rem; }
    .app-header h1 { font-size: 1.6rem; font-weight: 700; color: #1A2332; margin: 0; letter-spacing: -0.02em; }
    .app-header p  { font-size: 0.9rem; color: #5A6B7F; margin: 0.25rem 0 0 0; }
    .metric-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.25rem 1.5rem; }
    .metric-card .label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #5A6B7F; margin-bottom: 0.35rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
    .metric-card.teal .value { color: #0D7377; }
    .metric-card.red   .value { color: #D64545; }
    .metric-card.amber .value { color: #E8871E; }
    .metric-card.green .value { color: #2D8B4E; }
    .result-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; }
    .result-card .hospital-name { font-size: 1rem; font-weight: 600; color: #1A2332; }
    .result-card .hospital-meta { font-size: 0.8rem; color: #5A6B7F; margin-top: 0.2rem; }
    .hospital-data-row  { font-size: 0.85rem; color: #3D4F5F; margin-top: 0.3rem; line-height: 1.5; }
    .hospital-data-label { font-weight: 600; color: #0D7377; }
    .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
    .badge-high     { background: #D1FAE5; color: #065F46; }
    .badge-medium   { background: #FEF3C7; color: #92400E; }
    .badge-low      { background: #FEE2E2; color: #991B1B; }
    .badge-match    { background: #DBEAFE; color: #1E40AF; }
    .badge-critical { background: #FEE2E2; color: #991B1B; }
    .badge-high-risk{ background: #FEF3C7; color: #92400E; }
    .badge-moderate { background: #FEF9C3; color: #713F12; }
    .badge-adequate { background: #D1FAE5; color: #065F46; }
    .ai-answer { background: #F0FDFA; border: 1px solid #99F6E4; border-radius: 10px; padding: 1.25rem 1.5rem; margin-top: 1rem; font-size: 0.9rem; line-height: 1.6; color: #1A2332; }
    .ai-answer .answer-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #0D7377; margin-bottom: 0.5rem; }
    .search-context { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: #1E40AF; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1A2332; margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #0D7377; display: inline-block; }
    .gap-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; }
    .gap-table th { background: #F1F5F9; color: #475569; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; padding: 0.75rem 1rem; text-align: left; border-bottom: 2px solid #E2E8F0; }
    .gap-table td { padding: 0.65rem 1rem; border-bottom: 1px solid #F1F5F9; color: #334155; }
    .gap-table tr:hover td { background: #F8FAFC; }
    .anomaly-flag { background: #FFFBEB; border-left: 3px solid #F59E0B; padding: 0.5rem 0.75rem; margin-top: 0.5rem; font-size: 0.8rem; color: #92400E; border-radius: 0 6px 6px 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.9rem; color: #5A6B7F; padding: 0.75rem 1.25rem; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #0D7377 !important; border-bottom-color: #0D7377 !important; background: transparent !important; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;} .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════
df                    = load_data()
gap_df, services_dict = build_gap_analysis(df)
groq_key              = os.environ.get("GROQ_KEY", "")

# Sidebar
with st.sidebar:
    st.markdown("### Filters")
    if st.button("↻ Reset Filters", use_container_width=True):
        for k in ['sb_region','sb_type','sb_risk']:
            st.session_state.pop(k, None)
        st.rerun()

    regions         = sorted(df[df['region_clean'] != 'Unknown']['region_clean'].unique().tolist())
    selected_region = st.selectbox("Region", ["All"] + regions, key="sb_region")
    ftypes          = sorted(df[df['facilityTypeId'] != '']['facilityTypeId'].unique().tolist())
    selected_type   = st.selectbox("Facility type", ["All"] + ftypes, key="sb_type")
    selected_risk   = st.selectbox("Risk level", ['All','Critical','High Risk','Moderate','Adequate'], key="sb_risk")

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**{len(df)}** unique facilities")
    st.markdown(f"**{df[df['region_clean']!='Unknown']['region_clean'].nunique()}** regions with data")
    st.markdown(f"**{len(gap_df[gap_df['risk_level']=='Critical'])}** critical desert regions")

    # Show enrichment status
    has_enriched = 'enriched_capability' in df.columns and df['enriched_capability'].str.len().mean() > 10
    if has_enriched:
        icu_count = df['capability_text'].str.lower().str.contains('icu|intensive care', na=False).sum()
        st.markdown(f"**{icu_count}** hospitals with ICU data")
        st.markdown("🟢 Enriched capabilities active")
    else:
        st.markdown("🟡 Using base capabilities")

filtered_df = df.copy()
if selected_region != "All": filtered_df = filtered_df[filtered_df['region_clean'] == selected_region]
if selected_type   != "All": filtered_df = filtered_df[filtered_df['facilityTypeId'] == selected_type]

# Header
st.markdown(
    '<div class="app-header"><h1>⚕️ Ghana Healthcare Coverage</h1>'
    '<p>Identify medical deserts · Plan resource deployment · Powered by IDP Agent + Hybrid RAG</p></div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "🔍 Search", "📋 Regional Analysis", "🗺️ Map", "🏥 Directory"]
)

# ── TAB 1 DASHBOARD ──────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card teal"><div class="label">Total Facilities</div><div class="value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card red"><div class="label">Critical Deserts</div><div class="value">{len(gap_df[gap_df["risk_level"]=="Critical"])}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card amber"><div class="label">Services Tracked</div><div class="value">{len(services_dict)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card green"><div class="label">Regions Covered</div><div class="value">{df[df["region_clean"]!="Unknown"]["region_clean"].nunique()}</div></div>', unsafe_allow_html=True)

    import plotly.graph_objects as go
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.markdown('<div class="section-header">Facilities by Region</div>', unsafe_allow_html=True)
        rc = (filtered_df[filtered_df['region_clean'] != 'Unknown']
              .groupby('region_clean')['name'].count().reset_index())
        rc.columns = ['Region', 'Count']
        rc = rc.sort_values('Count', ascending=True)
        risk_map   = dict(zip(gap_df['region_clean'], gap_df['risk_level']))
        rc['Color'] = rc['Region'].map(risk_map).map(
            {'Critical':'#D64545','High Risk':'#E8871E','Moderate':'#D4A72C','Adequate':'#2D8B4E'}
        ).fillna('#94A3B8')
        fig = go.Figure(go.Bar(
            x=rc['Count'], y=rc['Region'], orientation='h',
            marker_color=rc['Color'], text=rc['Count'], textposition='outside'
        ))
        fig.update_layout(
            height=max(360, len(rc) * 28),
            margin=dict(l=0, r=40, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        rv = gap_df['risk_level'].value_counts().reset_index()
        rv.columns = ['Risk Level', 'Count']
        color_map = {'Critical':'#D64545','High Risk':'#E8871E','Moderate':'#D4A72C','Adequate':'#2D8B4E'}
        fig2 = go.Figure(go.Pie(
            labels=rv['Risk Level'], values=rv['Count'], hole=0.55,
            marker_colors=[color_map.get(r, '#94A3B8') for r in rv['Risk Level']]
        ))
        fig2.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2 SEARCH ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Search Healthcare Facilities</div>', unsafe_allow_html=True)
    st.caption("Powered by Hybrid BM25 + FAISS semantic search with enriched capabilities")

    query = st.text_input(
        "Search", placeholder="e.g. Which hospitals in Northern Ghana have emergency care?",
        label_visibility="collapsed"
    )

    if query:
        region_override = selected_region if selected_region != "All" else None
        is_gap_q        = is_gap_analysis_question(query)

        with st.spinner(f"Searching {len(df)} hospitals with hybrid RAG..."):
            results, detected_region, detected_service = hybrid_search(
                query, df, top_k=8, region_override=region_override
            )

        # Search context banner
        ctx_parts = []
        if detected_region:  ctx_parts.append(f"📍 Region: **{detected_region}**")
        if detected_service: ctx_parts.append(f"🏥 Service: **{detected_service}**")
        if not ctx_parts:    ctx_parts.append("🔍 Searching all regions and services")
        st.markdown(f'<div class="search-context">{" · ".join(ctx_parts)}</div>', unsafe_allow_html=True)

        # Gap analysis answer
        if is_gap_q:
            gap_answer = get_gap_analysis_answer(df)
            st.markdown(
                f'<div class="ai-answer"><div class="answer-label">📊 Gap Analysis</div>{gap_answer}</div>',
                unsafe_allow_html=True
            )
        elif groq_key:
            with st.spinner("Generating AI analysis..."):
                answer = get_llm_answer(query, results, groq_key, df, detected_region, detected_service)
            st.markdown(
                f'<div class="ai-answer"><div class="answer-label">🤖 AI Analysis</div>{answer}</div>',
                unsafe_allow_html=True
            )

        # Result cards
        relevant = [r for r in results if r.get('score_display', 0) > 0]
        others   = [r for r in results if r.get('score_display', 0) == 0]

        label = "Matching" if relevant else "Available"
        extra = f" + {len(others)} other" if others and relevant else ""
        st.markdown(
            f'<div class="section-header">{label} Facilities ({len(relevant)} relevant{extra})</div>',
            unsafe_allow_html=True
        )

        display_results = relevant + (others if not relevant else [])
        for r in display_results[:8]:
            anomalies    = detect_anomalies(r)
            anomaly_html = "".join(f'<div class="anomaly-flag">⚠️ {a}</div>' for a in anomalies)
            cap_html  = (f'<div class="hospital-data-row"><span class="hospital-data-label">Capabilities:</span> {r["capability"][:300]}</div>'
                         if r["capability"] else "")
            proc_html = (f'<div class="hospital-data-row"><span class="hospital-data-label">Procedures:</span> {r["procedure"][:200]}</div>'
                         if r["procedure"] else "")
            conf_class  = "high" if r["confidence"] == "High" else "medium" if r["confidence"] == "Medium" else "low"
            match_badge = (f'<span class="badge badge-match">✓ {detected_service} match</span>'
                           if r.get("score_display", 0) > 0 and detected_service else "")
            st.markdown(f"""<div class="result-card">
<div class="hospital-name">{r["name"]} <span class="badge badge-{conf_class}">{r["confidence"]} confidence</span> {match_badge}</div>
<div class="hospital-meta">📍 {r["region"]} · {r["city"]} · {r["type"]}</div>
{cap_html}{proc_html}{anomaly_html}
</div>""", unsafe_allow_html=True)

# ── TAB 3 REGIONAL ANALYSIS ──────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Regional Gap Analysis</div>', unsafe_allow_html=True)
    display_gap = gap_df[gap_df['risk_level'] == selected_risk] if selected_risk != "All" else gap_df
    table_html  = ('<table class="gap-table"><thead><tr>'
                   '<th>Region</th><th>Facilities</th><th>Services</th><th>Risk</th>'
                   + "".join(f'<th>{s}</th>' for s in services_dict)
                   + '</tr></thead><tbody>')
    for _, row in display_gap.iterrows():
        rc_ = row['risk_level'].lower().replace(' ', '-')
        table_html += (
            f'<tr><td><strong>{row["region_clean"]}</strong></td>'
            f'<td>{int(row["total_facilities"])}</td>'
            f'<td>{int(row["services_available"])}/8</td>'
            f'<td><span class="badge badge-{rc_}">{row["risk_level"]}</span></td>'
        )
        for svc in services_dict:
            has = row.get(f'has_{svc}', 0) > 0
            color = "#2D8B4E" if has else "#D64545"
            table_html += f'<td style="color:{color};font-weight:600;text-align:center">{"✓" if has else "✗"}</td>'
        table_html += '</tr>'
    st.markdown(table_html + '</tbody></table>', unsafe_allow_html=True)

# ── TAB 4 MAP ────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Medical Desert Map</div>', unsafe_allow_html=True)
    map_path = next((p for p in ["data/ghana_map.html", "ghana_map.html"] if os.path.exists(p)), None)
    if map_path:
        with open(map_path, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600, scrolling=False)
    else:
        st.warning("Map file not found. Place ghana_map.html in the data/ folder.")
        st.markdown("""
        **Legend:**
        - 🔴 Red = Critical Medical Desert (≤3 facilities)
        - 🟠 Orange = High Risk (≤8 facilities)
        - 🟡 Yellow = Moderate (≤20 facilities)
        - 🟢 Green = Adequate Coverage
        """)

# ── TAB 5 DIRECTORY ──────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Hospital Directory</div>', unsafe_allow_html=True)
    dir_search = st.text_input("Filter by name or city", placeholder="e.g. Korle Bu or Tamale")
    dir_df = filtered_df.copy()
    dir_df['address_city'] = dir_df['address_city'].fillna('').astype(str)
    if dir_search:
        mask = (
            dir_df['name'].str.lower().str.contains(dir_search.lower(), na=False) |
            dir_df['address_city'].str.lower().str.contains(dir_search.lower(), na=False)
        )
        dir_df = dir_df[mask]

    dcols = ['name', 'region_clean', 'address_city', 'facilityTypeId']
    for c in ['capability_text', 'procedure_text', 'specialties_text', 'confidence']:
        if c in dir_df.columns:
            dcols.append(c)

    st.dataframe(
        dir_df[dcols].rename(columns={
            'name':'Hospital', 'region_clean':'Region', 'address_city':'City',
            'facilityTypeId':'Type', 'capability_text':'Capabilities',
            'procedure_text':'Procedures', 'specialties_text':'Specialties',
            'confidence':'Confidence'
        }),
        use_container_width=True, hide_index=True, height=500
    )