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
    if " | " in str(val) and not str(val).startswith("["):
        items = [i.strip() for i in str(val).split(" | ") if i.strip()]
        clean = []
        for item in items:
            s = item.lower()
            if any(x in s for x in ['@', 'www.', 'http', '+233', 'located at',
                                     'contact', 'opening hours', 'closed on', 'facebook',
                                     'instagram', 'twitter']):
                continue
            clean.append(item)
        return " | ".join(clean)
    try:    parsed = json.loads(str(val))
    except:
        try:    parsed = ast.literal_eval(str(val))
        except: return str(val)
    if isinstance(parsed, list):
        clean = []
        for item in parsed:
            s = str(item).lower()
            if any(x in s for x in ['contact phone', 'email', '@', 'www.', 'located at',
                                     'currently closed', 'opening hours', 'closed on', 'http',
                                     'facebook', 'instagram', 'twitter', '+233']):
                continue
            clean.append(str(item).strip())
        return " | ".join(clean)
    return str(val)


def build_search_text(row):
    parts = [
        f"Hospital: {row.get('name', '')}",
        f"Region: {row.get('region_clean', '')}",
        f"Type: {row.get('facilityTypeId', '')}",
        f"City: {row.get('address_city', '')}",
    ]
    desc = str(row.get('description', '')).strip()
    if desc and desc not in ('nan', ''):
        parts.append(f"Description: {desc[:300]}")
    for label, col in [('Specialties', 'specialties_text'), ('Procedures', 'procedure_text'),
                       ('Capabilities', 'capability_text'), ('Equipment', 'equipment_text')]:
        v = str(row.get(col, '')).strip()
        if v and v != 'nan':
            parts.append(f"{label}: {v[:400]}")
    return " | ".join(parts)


# ════════════════════════════════════════════════════════════════
# REGION / SERVICE DETECTION
# ════════════════════════════════════════════════════════════════
REGION_MAP = {
    'Greater Accra': ['accra', 'greater accra', 'tema', 'legon', 'madina', 'spintex',
                      'cantonments', 'dome', 'ashaiman', 'east legon', 'dansoman',
                      'lapaz', 'achimota', 'nungua', 'adenta', 'ga east'],
    'Ashanti':       ['ashanti', 'kumasi', 'obuasi', 'ejisu', 'mampong', 'konongo'],
    'Northern':      ['northern', 'tamale', 'northern ghana', 'yendi', 'savelugu',
                      'kpandai', 'tatale', 'bimbilla', 'gushegu'],
    'Upper East':    ['upper east', 'bolgatanga', 'bawku', 'navrongo', 'zebilla', 'lamboya'],
    'Upper West':    ['upper west', 'wa ', ' wa,', 'lawra', 'jirapa', 'tumu', 'nandom'],
    'Volta':         ['volta', 'hohoe', 'aflao', 'keta', 'kpando', 'akatsi', 'ho ',
                      'sogakope', 'dzodze', 'battor', 'peki'],
    'Western':       ['western', 'takoradi', 'sekondi', 'tarkwa', 'axim', 'prestea'],
    'Central':       ['central', 'cape coast', 'winneba', 'elmina', 'saltpond', 'kasoa'],
    'Eastern':       ['eastern', 'koforidua', 'nkawkaw', 'akosombo', 'suhum', 'somanya'],
    'Brong Ahafo':   ['brong', 'sunyani', 'techiman', 'berekum', 'wenchi', 'dormaa'],
    'Savannah':      ['savannah', 'damongo', 'bole', 'salaga'],
    'North East':    ['north east', 'nalerigu', 'gambaga', 'bunkpurugu'],
    'Oti':           ['oti', 'dambai', 'nkwanta', 'worawora'],
    'Ahafo':         ['ahafo', 'goaso', 'bechem', 'kukuom'],
    'Western North': ['western north', 'bibiani', 'sefwi', 'juaboso', 'enchi'],
    'Bono East':     ['bono east', 'atebubu', 'ejura'],
}

SERVICE_MAP = {
    'icu':        ['icu', 'intensive care', 'critical care', 'nicu'],
    'emergency':  ['emergency', 'accident', '24/7', '24 hour', '24-hour', 'a&e', 'trauma'],
    'surgery':    ['surgery', 'surgical', 'operating theatre', 'operation', 'theatr'],
    'maternity':  ['maternity', 'obstetric', 'gynecol', 'gynaecol', 'delivery',
                   'labour', 'antenatal', 'prenatal'],
    'laboratory': ['laboratory', 'lab test', 'diagnostic lab', 'pathology', 'blood test'],
    'imaging':    ['mri', 'ct scan', 'x-ray', 'xray', 'imaging', 'radiology',
                   'ultrasound', 'mammogram'],
    'pediatrics': ['pediatric', 'paediatric', 'children hospital', 'neonatal', 'child care'],
    'pharmacy':   ['pharmacy', 'pharmacist', 'dispensary', 'pharmaceutical'],
    'dental':     ['dental', 'dentist', 'dentistry', 'orthodon'],
    'cardiac':    ['cardiac', 'cardiology', 'heart surgery', 'echocardiography'],
    'dialysis':   ['dialysis', 'renal', 'kidney', 'nephrology'],
    'eye':        ['ophthalmology', 'eye', 'vision', 'cataract', 'glaucoma', 'retina'],
}

QUERY_SYNONYMS = {
    "icu":        ["icu", "intensive care unit", "critical care", "intensive care"],
    "emergency":  ["emergency", "accident and emergency", "trauma", "24 hour", "urgent care"],
    "surgery":    ["surgery", "surgical", "operating theatre", "operation"],
    "maternity":  ["maternity", "obstetric", "delivery", "labour", "antenatal"],
    "pediatric":  ["pediatric", "children", "neonatal", "child health", "paediatric"],
    "imaging":    ["imaging", "x-ray", "xray", "mri", "ct scan", "ultrasound", "radiology"],
    "laboratory": ["laboratory", "lab", "diagnostic testing", "pathology"],
    "pharmacy":   ["pharmacy", "pharmaceutical", "dispensary"],
    "dental":     ["dental", "dentistry", "tooth"],
    "dialysis":   ["dialysis", "renal", "kidney", "nephrology"],
    "deploy":     ["medical desert", "underserved", "few hospitals", "urgent", "shortage"],
    "desert":     ["medical desert", "underserved", "scarce", "limited hospitals"],
}

GAP_TRIGGERS = ['desert', 'deploy', 'few hospital', 'most urgently', 'underserved',
                'shortage', 'where should', 'which region', 'least hospital',
                'coverage', 'how many region']

NGO_TRIGGERS = ['ngo', 'foundation', 'charity', 'volunteer', 'aid organisation',
                'non-profit', 'nonprofit', 'community health program']


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
    return any(x in q for x in GAP_TRIGGERS)


def is_ngo_question(query):
    q = query.lower()
    return any(x in q for x in NGO_TRIGGERS)


# ════════════════════════════════════════════════════════════════
# DATA LOADING
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

    # Use enriched_capability if available (from Databricks IDP pipeline)
    has_enriched = (
        'enriched_capability' in df.columns and
        df['enriched_capability'].astype(str).str.len().mean() > 10
    )

    df['procedure_text']   = df['procedure'].apply(unpack_list)   if 'procedure'  in df.columns else ""
    df['equipment_text']   = df['equipment'].apply(unpack_list)   if 'equipment'  in df.columns else ""
    df['specialties_text'] = df['specialties'].apply(unpack_list) if 'specialties' in df.columns else ""

    if has_enriched:
        df['capability_text'] = df['enriched_capability'].apply(clean_capability)
    elif 'capability_text' in df.columns:
        df['capability_text'] = df['capability_text'].apply(clean_capability)
    elif 'capability' in df.columns:
        df['capability_text'] = df['capability'].apply(clean_capability)
    else:
        df['capability_text'] = ""

    df['search_text_full'] = df.apply(build_search_text, axis=1)

    # Dedup — keep most complete row per hospital name
    def completeness(row):
        return sum(1 for c in ['procedure_text', 'equipment_text', 'capability_text',
                               'specialties_text', 'address_city']
                   if str(row.get(c, '')).strip() not in ['', 'nan'])

    df['_c'] = df.apply(completeness, axis=1)
    df = df.sort_values('_c', ascending=False).drop_duplicates(subset=['name'], keep='first')
    df = df.drop(columns=['_c']).reset_index(drop=True)

    # Region normalization
    region_fixes = {
        'Accra East': 'Greater Accra', 'Accra North': 'Greater Accra',
        'Ga East Municipality': 'Greater Accra',
        'Ga East Municipality, Greater Accra Region': 'Greater Accra',
        'Ledzokuku-Krowor': 'Greater Accra', 'Kpone Katamanso': 'Greater Accra',
        'Shai Osudoku District, Greater Accra Region': 'Greater Accra',
        'Tema West Municipal': 'Greater Accra',
        'Asokwa-Kumasi': 'Ashanti', 'Ejisu Municipal': 'Ashanti',
        'Ahafo Ano South-East': 'Ashanti', 'SH': 'Ashanti',
        'KEEA': 'Central', 'Central Ghana': 'Central',
        'Central Tongu District': 'Volta',
        'Asutifi South': 'Ahafo', 'Bono': 'Brong Ahafo',
        'Dormaa East': 'Brong Ahafo', 'Techiman Municipal': 'Brong Ahafo',
        'Takoradi': 'Western', 'Sissala West District': 'Upper West',
        'Ghana': 'Unknown',
    }
    df['region_clean'] = df['region_clean'].replace(region_fixes)
    df.loc[df['region_clean'].isin(['', 'nan', 'Unknown']), 'region_clean'] = 'Unknown'

    # Fix unknown regions by city name
    city_to_region = {
        'accra': 'Greater Accra', 'tema': 'Greater Accra', 'madina': 'Greater Accra',
        'legon': 'Greater Accra', 'adenta': 'Greater Accra', 'dansoman': 'Greater Accra',
        'lapaz': 'Greater Accra', 'achimota': 'Greater Accra', 'nungua': 'Greater Accra',
        'weija': 'Greater Accra', 'ashaiman': 'Greater Accra', 'amasaman': 'Greater Accra',
        'cantonments': 'Greater Accra', 'east legon': 'Greater Accra', 'dome': 'Greater Accra',
        'kumasi': 'Ashanti', 'obuasi': 'Ashanti', 'ejisu': 'Ashanti',
        'takoradi': 'Western', 'sekondi': 'Western', 'tarkwa': 'Western',
        'cape coast': 'Central', 'winneba': 'Central', 'saltpond': 'Central',
        'tamale': 'Northern', 'yendi': 'Northern', 'savelugu': 'Northern',
        'bolgatanga': 'Upper East', 'bawku': 'Upper East', 'navrongo': 'Upper East',
        'wa': 'Upper West', 'lawra': 'Upper West',
        'hohoe': 'Volta', 'kpando': 'Volta', 'aflao': 'Volta',
        'sunyani': 'Brong Ahafo', 'techiman': 'Brong Ahafo',
        'akosombo': 'Eastern', 'koforidua': 'Eastern',
        'sefwi': 'Western North', 'bibiani': 'Western North',
        'damongo': 'Savannah', 'nalerigu': 'North East', 'gambaga': 'North East',
        'dambai': 'Oti', 'nkwanta': 'Oti', 'worawora': 'Oti',
        'atebubu': 'Bono East', 'ejura': 'Bono East',
        'goaso': 'Ahafo', 'bechem': 'Ahafo',
    }

    def fix_unknown(row):
        if row['region_clean'] != 'Unknown':
            return row['region_clean']
        combined = (str(row.get('address_city', '')).lower() + ' ' +
                    str(row.get('name', '')).lower())
        for k, v in city_to_region.items():
            if k in combined:
                return v
        return 'Unknown'

    df['region_clean'] = df.apply(fix_unknown, axis=1)

    # Add confidence if missing
    if 'confidence' not in df.columns:
        df['confidence'] = 'Medium'

    return df


# ════════════════════════════════════════════════════════════════
# HYBRID SEARCH INDEX
# FIX: load new faiss_index.bin first, fallback to old files
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_search_index(_df_len, _df_cols):
    df = st.session_state.get("_search_df")
    if df is None:
        return None, None, None, None, []

    # Medical synonym expansion — same as evaluation notebook
    MEDICAL_SYNONYMS = {
        "icu":        "ICU intensive care unit critical care level-3",
        "emergency":  "emergency A&E accident trauma 24-hour resuscitation",
        "maternity":  "maternity obstetrics delivery labour antenatal prenatal",
        "surgery":    "surgery surgical theatre operating room procedures",
        "pediatric":  "pediatric paediatric children child neonatal NICU",
        "imaging":    "imaging MRI CT scan X-ray radiology ultrasound",
        "laboratory": "laboratory lab diagnostic pathology blood test",
        "dialysis":   "dialysis renal kidney nephrology",
    }

    def build_rich_text(row):
        parts = [
            f"Hospital: {row.get('name', '')}",
            f"Region: {row.get('region_clean', '')}",
            f"City: {row.get('address_city', '')}",
            f"Type: {row.get('facilityTypeId', '')}",
        ]
        desc = str(row.get('description', '')).strip()
        if desc and desc != 'nan':
            parts.append(f"Description: {desc[:300]}")

        specs = str(row.get('specialties_text', '')).strip()
        if specs and specs != 'nan':
            parts.append(f"Specialties: {specs}")
            parts.append(f"Specialties: {specs}")

        cap = str(row.get('capability_text', '')).strip()
        combined = ""
        if cap and cap != 'nan':
            parts.append(f"Capabilities: {cap}")
            parts.append(f"Capabilities: {cap}")
            parts.append(f"Capabilities: {cap}")
            combined += cap.lower()

        proc = str(row.get('procedure_text', '')).strip()
        if proc and proc != 'nan':
            parts.append(f"Procedures: {proc}")
            parts.append(f"Procedures: {proc}")
            combined += " " + proc.lower()

        # Add synonym expansions
        for concept, synonyms in MEDICAL_SYNONYMS.items():
            triggers = [concept] + synonyms.lower().split()[:2]
            if any(t in combined for t in triggers):
                parts.append(synonyms)

        return " | ".join(parts)

    texts = df.apply(build_rich_text, axis=1).tolist()

    # BM25
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    # FAISS — try new index first, then old
    import faiss as faiss_lib
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # FIX: prioritise new faiss_index.bin over old hospital_index.faiss
    faiss_path = next((p for p in [
        "data/faiss_index.bin", "faiss_index.bin",
        "data/hospital_index.faiss", "hospital_index.faiss",
    ] if os.path.exists(p)), None)

    emb_path = next((p for p in [
        "data/embeddings.npy", "embeddings.npy",
        "data/hospital_embeddings.npy", "hospital_embeddings.npy",
    ] if os.path.exists(p)), None)

    if faiss_path and emb_path:
        try:
            faiss_index = faiss_lib.read_index(faiss_path)
            embeddings  = np.load(emb_path)
            # Rebuild if size mismatch (e.g. old index has different row count)
            if faiss_index.ntotal != len(df):
                raise ValueError(f"Index size {faiss_index.ntotal} != df size {len(df)}")
        except Exception:
            embeddings  = embedder.encode(texts, show_progress_bar=False, batch_size=64).astype('float32')
            faiss_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)
    else:
        embeddings  = embedder.encode(texts, show_progress_bar=False, batch_size=64).astype('float32')
        faiss_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)

    return bm25, faiss_index, embeddings, embedder, texts


def hybrid_search(query, df, top_k=8, region_override=None):
    region  = region_override or detect_region(query)
    service, svc_keywords = detect_service(query)

    st.session_state["_search_df"] = df
    bm25, faiss_index, embeddings, embedder, texts = load_search_index(
        len(df), str(df.columns.tolist())
    )

    if bm25 is None:
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

    # FIX: soft region boost instead of hard filter
    # Hospitals with null region_clean but correct city are now boosted, not excluded
    if region:
        region_kws = REGION_MAP.get(region, [region.lower()])
        for i in range(len(df)):
            row = df.iloc[i]
            loc = (str(row.get('region_clean', '')) + ' ' +
                   str(row.get('address_city', ''))).lower()
            if any(k in loc for k in region_kws):
                hybrid_scores[i] *= 1.8

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

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Mark service matches
    for r in results:
        all_text = (r['capability'] + ' ' + r['procedure'] + ' ' + r['specialties']).lower()
        r['score_display'] = int(
            bool(svc_keywords) and any(k in all_text for k in svc_keywords)
        )

    return results[:top_k], region, service


def _keyword_fallback(query, df, region, service, svc_keywords, top_k):
    pool = df.copy()
    if region:
        region_kws = REGION_MAP.get(region, [])
        mask = pool.apply(
            lambda r: any(k in (str(r['region_clean']) + ' ' +
                                str(r.get('address_city', ''))).lower()
                          for k in region_kws), axis=1
        )
        filtered = pool[mask]
        if len(filtered) > 0:
            pool = filtered

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
# ════════════════════════════════════════════════════════════════
@st.cache_data
def build_gap_analysis(df):
    services = {
        'ICU':        ['icu', 'intensive care', 'critical care'],
        'Emergency':  ['emergency', 'accident', '24/7', '24 hour'],
        'Surgery':    ['surgery', 'surgical', 'operating theatre'],
        'Maternity':  ['maternity', 'obstetric', 'gynecology', 'delivery', 'gynaecology'],
        'Laboratory': ['laboratory', 'lab test', 'diagnostic lab'],
        'Imaging':    ['x-ray', 'xray', 'ultrasound', 'mri', 'ct scan', 'radiology'],
        'Pediatrics': ['pediatric', 'paediatric', 'children', 'neonatal'],
        'Pharmacy':   ['pharmacy', 'pharmaceutical', 'dispensary'],
    }
    df2 = df.copy()
    for svc, kws in services.items():
        df2[f'has_{svc}'] = df2['search_text_full'].str.lower().apply(
            lambda t: any(k in t for k in kws)
        )

    svc_cols = [f'has_{s}' for s in services]
    rs = (df2[df2['region_clean'] != 'Unknown']
          .groupby('region_clean')
          .agg(total_facilities=('name', 'count'),
               **{c: (c, 'sum') for c in svc_cols})
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
    region_counts = (df[df['region_clean'] != 'Unknown']['region_clean']
                     .value_counts().sort_values())
    lines = ["**Medical desert regions in Ghana (fewest hospitals):**\n"]
    for region, count in region_counts.items():
        risk = ('🔴 CRITICAL' if count <= 3 else
                '🟠 HIGH RISK' if count <= 8 else
                '🟡 MODERATE'  if count <= 20 else '🟢 ADEQUATE')
        rd  = df[df['region_clean'] == region]
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
    all_text = (result.get('procedure', '') + ' ' +
                result.get('capability', '') + ' ' +
                result.get('equipment', '')).lower()
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
    for i, r in enumerate(results[:5], 1):
        context += (
            f"Hospital {i}: {r['name']} | Region: {r['region']} | City: {r['city']}\n"
            f"  Capabilities: {r['capability'][:250]}\n"
            f"  Procedures  : {r['procedure'][:150]}\n"
            f"  Specialties : {r['specialties'][:150]}\n\n"
        )

    prompt = f"""You are a healthcare analyst for Virtue Foundation in Ghana.

DATABASE STATS:
{stats}

RETRIEVED HOSPITALS (filtered to {region or 'all regions'}{f', sorted by {service} relevance' if service else ''}):
{context}

QUESTION: {question}

Answer rules:
- Name specific hospitals from the list above with their region/city.
- If asked about ICU: look for 'ICU', 'intensive care' in capabilities.
- If asked about a specific region: ONLY mention hospitals from that region.
- If 0 relevant hospitals found for the service, say so honestly.
- Keep answer under 200 words, factual and specific."""

    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]:
        try:
            resp = gclient.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, n=1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                import time
                time.sleep(10)
                continue
            continue
    return "⚠️ AI model temporarily unavailable. Showing retrieved hospital data above."


# ════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ghana Healthcare Coverage · Virtue Foundation",
    page_icon="⚕️", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; background-color: #F4F6F9; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0D2137 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    section[data-testid="stSidebar"] h3 { color: #F8FAFC !important; font-size: 0.75rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #94A3B8 !important; font-size: 0.75rem !important; }
    section[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; }

    .app-header {
        background: linear-gradient(135deg, #0A1628 0%, #0D3B6E 50%, #1A5276 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute; top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(13,147,197,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem; font-weight: 400;
        color: #F8FAFC; margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p { font-size: 0.9rem; color: #94A3B8; margin: 0.4rem 0 0 0; }
    .app-header .eval-badge {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem; font-weight: 600;
        color: #34D399;
        margin-top: 0.75rem;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    .metric-card .label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; margin-bottom: 0.4rem; }
    .metric-card .value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
    .metric-card .sub   { font-size: 0.78rem; color: #94A3B8; margin-top: 0.25rem; }
    .metric-card.teal .value  { color: #0891B2; }
    .metric-card.red .value   { color: #DC2626; }
    .metric-card.amber .value { color: #D97706; }
    .metric-card.green .value { color: #059669; }
    .metric-card.purple .value{ color: #7C3AED; }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #E8EDF3;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
    .result-card.match-card { border-left: 3px solid #0891B2; }
    .hospital-name { font-size: 1rem; font-weight: 600; color: #0F172A; }
    .hospital-meta { font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }
    .hospital-data-row  { font-size: 0.83rem; color: #334155; margin-top: 0.4rem; line-height: 1.6; }
    .hospital-data-label { font-weight: 600; color: #0891B2; margin-right: 0.25rem; }

    .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.02em; margin-left: 0.25rem; }
    .badge-high     { background: #DCFCE7; color: #166534; }
    .badge-medium   { background: #FEF9C3; color: #854D0E; }
    .badge-low      { background: #FEE2E2; color: #991B1B; }
    .badge-match    { background: #DBEAFE; color: #1D4ED8; }
    .badge-critical { background: #FEE2E2; color: #991B1B; }
    .badge-high-risk{ background: #FEF3C7; color: #92400E; }
    .badge-moderate { background: #FEF9C3; color: #713F12; }
    .badge-adequate { background: #DCFCE7; color: #166534; }
    .badge-ngo      { background: #F3E8FF; color: #6B21A8; }

    .ai-answer {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #0F172A;
    }
    .ai-answer .answer-label {
        font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: #0369A1; margin-bottom: 0.6rem;
    }

    .search-context {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 1rem; font-size: 0.82rem; color: #1D4ED8;
    }

    .section-header {
        font-size: 1rem; font-weight: 700; color: #0F172A;
        margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #0891B2; display: inline-block;
        letter-spacing: -0.01em;
    }

    .gap-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.83rem; }
    .gap-table th {
        background: #F8FAFC; color: #475569; font-weight: 700;
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
        padding: 0.75rem 1rem; text-align: left; border-bottom: 2px solid #E2E8F0;
    }
    .gap-table td { padding: 0.65rem 1rem; border-bottom: 1px solid #F1F5F9; color: #334155; }
    .gap-table tr:hover td { background: #F8FAFC; }

    .anomaly-flag {
        background: #FFFBEB; border-left: 3px solid #F59E0B;
        padding: 0.4rem 0.75rem; margin-top: 0.5rem;
        font-size: 0.78rem; color: #92400E;
        border-radius: 0 6px 6px 0;
    }

    .eval-panel {
        background: linear-gradient(135deg, #0A1628, #0D3B6E);
        border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.5rem;
    }
    .eval-panel .eval-title {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 0.5rem;
    }
    .eval-row { display: flex; justify-content: space-between; font-size: 0.78rem; margin: 0.2rem 0; }
    .eval-row .metric-name { color: #94A3B8; }
    .eval-row .metric-val  { color: #34D399; font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #E2E8F0; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif; font-weight: 500;
        font-size: 0.88rem; color: #64748B;
        padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent; margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] { color: #0891B2 !important; border-bottom-color: #0891B2 !important; background: transparent !important; font-weight: 700 !important; }

    div[data-testid="stTextInput"] input {
        border-radius: 10px; border: 2px solid #E2E8F0;
        font-size: 0.95rem; padding: 0.75rem 1rem;
        font-family: 'DM Sans', sans-serif;
        transition: border-color 0.2s;
    }
    div[data-testid="stTextInput"] input:focus { border-color: #0891B2; box-shadow: 0 0 0 3px rgba(8,145,178,0.1); }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    header {visibility:hidden;} .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════
df                    = load_data()
gap_df, services_dict = build_gap_analysis(df)
groq_key              = os.environ.get("GROQ_KEY", "")

# Try secrets
if not groq_key:
    try:
        groq_key = st.secrets.get("GROQ_KEY", "")
    except Exception:
        pass

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚕️ Ghana Healthcare")

    st.markdown("---")

    # Eval score panel
    has_enriched = ('enriched_capability' in df.columns and
                    df['enriched_capability'].astype(str).str.len().mean() > 10)
    icu_count = df['capability_text'].str.lower().str.contains(
        'icu|intensive care', na=False).sum() if 'capability_text' in df.columns else 0

    st.markdown(f"""
<div class="eval-panel">
  <div class="eval-title">🧪 RAG Evaluation Score</div>
  <div class="eval-row"><span class="metric-name">Overall Score</span><span class="metric-val">0.923 / 1.0</span></div>
  <div class="eval-row"><span class="metric-name">Questions Passed</span><span class="metric-val">12 / 12 ✅</span></div>
  <div class="eval-row"><span class="metric-name">Answer Correctness</span><span class="metric-val">0.958</span></div>
  <div class="eval-row"><span class="metric-name">Retrieval Quality</span><span class="metric-val">0.909</span></div>
  <div class="eval-row"><span class="metric-name">Search Type</span><span class="metric-val">Hybrid BM25+FAISS</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Filters")

    if st.button("↻ Reset Filters", use_container_width=True):
        for k in ['sb_region', 'sb_type', 'sb_risk']:
            st.session_state.pop(k, None)
        st.rerun()

    regions         = sorted(df[df['region_clean'] != 'Unknown']['region_clean'].unique().tolist())
    selected_region = st.selectbox("Region", ["All"] + regions, key="sb_region")
    ftypes          = sorted([x for x in df['facilityTypeId'].unique().tolist() if x])
    selected_type   = st.selectbox("Facility type", ["All"] + ftypes, key="sb_type")
    selected_risk   = st.selectbox("Risk level", ['All', 'Critical', 'High Risk', 'Moderate', 'Adequate'], key="sb_risk")

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**{len(df)}** unique facilities")
    st.markdown(f"**{df[df['region_clean']!='Unknown']['region_clean'].nunique()}** regions with data")
    st.markdown(f"**{len(gap_df[gap_df['risk_level']=='Critical'])}** critical desert regions")
    if has_enriched:
        st.markdown(f"**{icu_count}** hospitals with ICU data")
        st.markdown("🟢 Enriched capabilities active")
    else:
        st.markdown("🟡 Using base capabilities")

filtered_df = df.copy()
if selected_region != "All": filtered_df = filtered_df[filtered_df['region_clean'] == selected_region]
if selected_type   != "All": filtered_df = filtered_df[filtered_df['facilityTypeId'] == selected_type]

# ── HEADER ───────────────────────────────────────────────────
critical_count = len(gap_df[gap_df['risk_level'] == 'Critical'])
st.markdown(f"""
<div class="app-header">
  <h1>⚕️ Ghana Healthcare Coverage</h1>
  <p>Identify medical deserts · Plan resource deployment · Powered by IDP Agent + Hybrid RAG</p>
  <span class="eval-badge">✅ RAG Score: 0.923 · 12/12 questions passed · 896 hospitals indexed</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "🔍 Search", "📋 Regional Analysis", "🗺️ Map", "🏥 Directory"]
)

# ════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "teal",   "Total Facilities",  len(filtered_df),       "in database"),
        (c2, "red",    "Critical Deserts",  critical_count,          "regions"),
        (c3, "amber",  "Services Tracked",  len(services_dict),     "types"),
        (c4, "green",  "Regions Covered",   df[df['region_clean']!='Unknown']['region_clean'].nunique(), "regions"),
        (c5, "purple", "RAG Score",         "0.923",                "12/12 passed"),
    ]
    for col, cls, label, val, sub in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card {cls}">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{val}</div>'
                f'<div class="sub">{sub}</div>'
                f'</div>', unsafe_allow_html=True
            )

    import plotly.graph_objects as go
    st.markdown("")
    cc1, cc2 = st.columns([3, 2])

    with cc1:
        st.markdown('<div class="section-header">Facilities by Region</div>', unsafe_allow_html=True)
        rc = (filtered_df[filtered_df['region_clean'] != 'Unknown']
              .groupby('region_clean')['name'].count().reset_index())
        rc.columns = ['Region', 'Count']
        rc = rc.sort_values('Count', ascending=True)
        risk_map  = dict(zip(gap_df['region_clean'], gap_df['risk_level']))
        rc['Color'] = rc['Region'].map(risk_map).map({
            'Critical': '#DC2626', 'High Risk': '#D97706',
            'Moderate': '#CA8A04', 'Adequate': '#059669'
        }).fillna('#94A3B8')
        fig = go.Figure(go.Bar(
            x=rc['Count'], y=rc['Region'], orientation='h',
            marker_color=rc['Color'],
            text=rc['Count'], textposition='outside',
            hovertemplate='<b>%{y}</b><br>Facilities: %{x}<extra></extra>'
        ))
        fig.update_layout(
            height=max(380, len(rc) * 26),
            margin=dict(l=0, r=50, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False),
            yaxis=dict(showgrid=False),
            font=dict(family='DM Sans', size=12, color='#334155'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        rv = gap_df['risk_level'].value_counts().reset_index()
        rv.columns = ['Risk Level', 'Count']
        color_map = {
            'Critical': '#DC2626', 'High Risk': '#D97706',
            'Moderate': '#CA8A04', 'Adequate': '#059669'
        }
        fig2 = go.Figure(go.Pie(
            labels=rv['Risk Level'], values=rv['Count'], hole=0.6,
            marker_colors=[color_map.get(r, '#94A3B8') for r in rv['Risk Level']],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Regions: %{value}<extra></extra>'
        ))
        fig2.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family='DM Sans', size=12),
            annotations=[dict(text=f'<b>{len(gap_df)}</b><br>Regions',
                             x=0.5, y=0.5, font_size=14,
                             font_color='#0F172A', showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Service coverage summary
        st.markdown('<div class="section-header">Service Coverage</div>', unsafe_allow_html=True)
        for svc in services_dict:
            col_name = f'has_{svc}'
            if col_name in gap_df.columns:
                covered = int((gap_df[col_name] > 0).sum())
                total_r = len(gap_df)
                pct = covered / total_r * 100
                color = "#059669" if pct > 60 else "#D97706" if pct > 30 else "#DC2626"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.8rem;padding:0.2rem 0;border-bottom:1px solid #F1F5F9;">'
                    f'<span style="color:#475569">{svc}</span>'
                    f'<span style="color:{color};font-weight:700">{covered}/{total_r} regions</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Search Healthcare Facilities</div>', unsafe_allow_html=True)
    st.caption("Hybrid BM25 + FAISS semantic search · Medical synonym expansion · Region boost · 896 hospitals indexed")

    # Example queries
    col_a, col_b, col_c, col_d = st.columns(4)
    example_queries = [
        "ICU hospitals in Accra",
        "Emergency care in Northern Ghana",
        "Maternity services Volta region",
        "MRI or CT scan imaging Ghana",
    ]
    for col, eq in zip([col_a, col_b, col_c, col_d], example_queries):
        with col:
            if st.button(eq, use_container_width=True):
                st.session_state['search_query'] = eq

    default_q = st.session_state.get('search_query', '')
    query = st.text_input(
        "Search",
        value=default_q,
        placeholder="e.g. Which hospitals in Northern Ghana have emergency care?",
        label_visibility="collapsed"
    )

    if query:
        region_override = selected_region if selected_region != "All" else None
        is_gap_q = is_gap_analysis_question(query)
        is_ngo_q = is_ngo_question(query)

        with st.spinner(f"Searching {len(df)} hospitals with hybrid RAG..."):
            results, detected_region, detected_service = hybrid_search(
                query, df, top_k=8, region_override=region_override
            )

        # Search context banner
        ctx_parts = []
        if detected_region:  ctx_parts.append(f"📍 Region: **{detected_region}**")
        if detected_service: ctx_parts.append(f"🏥 Service: **{detected_service}**")
        if is_ngo_q:         ctx_parts.append("🤝 NGO search active")
        if not ctx_parts:    ctx_parts.append("🔍 Searching all regions and services")
        st.markdown(
            f'<div class="search-context">{" · ".join(ctx_parts)}</div>',
            unsafe_allow_html=True
        )

        # Gap analysis answer (pre-computed, no LLM needed)
        if is_gap_q:
            gap_answer_text = get_gap_analysis_answer(df)
            st.markdown(
                f'<div class="ai-answer">'
                f'<div class="answer-label">📊 Gap Analysis (Pre-computed from 896 hospitals)</div>'
                f'{gap_answer_text}'
                f'</div>',
                unsafe_allow_html=True
            )
        elif groq_key:
            with st.spinner("Generating AI analysis..."):
                answer = get_llm_answer(
                    query, results, groq_key, df, detected_region, detected_service
                )
            st.markdown(
                f'<div class="ai-answer">'
                f'<div class="answer-label">🤖 AI Analysis (llama-3.3-70b)</div>'
                f'{answer}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("💡 Add GROQ_KEY to Streamlit Secrets to enable AI-powered answers.")

        # Result cards
        relevant = [r for r in results if r.get('score_display', 0) > 0]
        others   = [r for r in results if r.get('score_display', 0) == 0]

        label = "Matching" if relevant else "Available"
        extra = f" + {len(others)} other hospitals" if others and relevant else ""
        st.markdown(
            f'<div class="section-header">{label} Facilities '
            f'({len(relevant)} relevant{extra})</div>',
            unsafe_allow_html=True
        )

        display_results = (relevant + others) if relevant else results
        for r in display_results[:8]:
            anomalies    = detect_anomalies(r)
            anomaly_html = "".join(f'<div class="anomaly-flag">⚠️ {a}</div>' for a in anomalies)

            cap_html  = (f'<div class="hospital-data-row">'
                         f'<span class="hospital-data-label">Capabilities:</span>'
                         f' {r["capability"][:350]}</div>'
                         if r["capability"] and r["capability"] != 'nan' else "")
            proc_html = (f'<div class="hospital-data-row">'
                         f'<span class="hospital-data-label">Procedures:</span>'
                         f' {r["procedure"][:200]}</div>'
                         if r["procedure"] and r["procedure"] != 'nan' else "")
            spec_html = (f'<div class="hospital-data-row">'
                         f'<span class="hospital-data-label">Specialties:</span>'
                         f' {r["specialties"][:200]}</div>'
                         if r["specialties"] and r["specialties"] != 'nan' else "")

            conf      = r["confidence"].lower() if r["confidence"] else "medium"
            conf_cls  = "high" if "high" in conf else "medium" if "medium" in conf else "low"
            match_badge = (f'<span class="badge badge-match">✓ {detected_service}</span>'
                           if r.get("score_display", 0) > 0 and detected_service else "")
            card_class = "result-card match-card" if r.get("score_display", 0) > 0 else "result-card"
            ftype = str(r.get("type", "")).lower()
            type_badge = (f'<span class="badge badge-ngo">NGO</span>'
                          if ftype == 'ngo' else "")

            st.markdown(f"""
<div class="{card_class}">
  <div class="hospital-name">
    {r["name"]}
    <span class="badge badge-{conf_cls}">{r["confidence"]} confidence</span>
    {match_badge}{type_badge}
  </div>
  <div class="hospital-meta">
    📍 {r["region"] or "Unknown region"} · {r["city"] or "Unknown city"} · {r["type"] or "facility"}
  </div>
  {cap_html}{proc_html}{spec_html}{anomaly_html}
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Regional Gap Analysis</div>', unsafe_allow_html=True)
    st.caption("Based on enriched capability data from 896 Ghana healthcare facilities")

    # Summary stats
    s1, s2, s3, s4 = st.columns(4)
    risk_counts = gap_df['risk_level'].value_counts()
    for col, risk, cls in [(s1,'Critical','red'),(s2,'High Risk','amber'),
                            (s3,'Moderate','teal'),(s4,'Adequate','green')]:
        with col:
            cnt = risk_counts.get(risk, 0)
            st.markdown(
                f'<div class="metric-card {cls}">'
                f'<div class="label">{risk}</div>'
                f'<div class="value">{cnt}</div>'
                f'<div class="sub">regions</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("")

    # Filter gap table
    display_gap = gap_df.copy()
    if selected_risk != 'All':
        display_gap = display_gap[display_gap['risk_level'] == selected_risk]
    if selected_region != 'All':
        display_gap = display_gap[display_gap['region_clean'] == selected_region]

    table_html = (
        '<table class="gap-table"><thead><tr>'
        '<th>Region</th><th>Facilities</th><th>Services</th><th>Risk Level</th>'
        + "".join(f'<th>{s}</th>' for s in services_dict)
        + '</tr></thead><tbody>'
    )
    for _, row in display_gap.iterrows():
        rc_ = row['risk_level'].lower().replace(' ', '-')
        table_html += (
            f'<tr><td><strong>{row["region_clean"]}</strong></td>'
            f'<td style="text-align:center">{int(row["total_facilities"])}</td>'
            f'<td style="text-align:center">{int(row["services_available"])}/8</td>'
            f'<td><span class="badge badge-{rc_}">{row["risk_level"]}</span></td>'
        )
        for svc in services_dict:
            has = row.get(f'has_{svc}', 0) > 0
            color = "#059669" if has else "#DC2626"
            icon  = "✓" if has else "✗"
            table_html += (f'<td style="color:{color};font-weight:700;'
                           f'text-align:center">{icon}</td>')
        table_html += '</tr>'
    st.markdown(table_html + '</tbody></table>', unsafe_allow_html=True)

    # Deployment recommendation
    st.markdown("")
    st.markdown('<div class="section-header">Deployment Recommendations</div>', unsafe_allow_html=True)
    critical_regions = gap_df[gap_df['risk_level'] == 'Critical'].sort_values('total_facilities')
    if len(critical_regions) > 0:
        for _, row in critical_regions.iterrows():
            missing_svcs = [s for s in services_dict
                            if not row.get(f'has_{s}', False)]
            st.markdown(
                f'<div class="anomaly-flag" style="border-color:#DC2626">'
                f'🔴 <strong>{row["region_clean"]}</strong> — '
                f'{int(row["total_facilities"])} facilities — '
                f'Missing: {", ".join(missing_svcs[:4])}'
                f'</div>',
                unsafe_allow_html=True
            )

# ════════════════════════════════════════════════════════════════
# TAB 4 — MAP
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Medical Desert Map</div>', unsafe_allow_html=True)
    st.caption("Color-coded by risk level: 🔴 Critical · 🟠 High Risk · 🟡 Moderate · 🟢 Adequate")

    map_path = next(
        (p for p in ["data/ghana_map.html", "ghana_map.html"] if os.path.exists(p)),
        None
    )
    if map_path:
        with open(map_path, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600, scrolling=False)
    else:
        st.warning("Map file not found. Place ghana_map.html in the data/ folder.")
        # Show a table fallback
        st.markdown("**Regional distribution (map unavailable):**")
        map_data = (df[df['region_clean'] != 'Unknown']
                    .groupby('region_clean')['name']
                    .count().reset_index()
                    .rename(columns={'name': 'Facilities', 'region_clean': 'Region'})
                    .sort_values('Facilities'))
        st.dataframe(map_data, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — DIRECTORY
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Hospital Directory</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered_df)} facilities (use sidebar filters to narrow down)")

    dir_search = st.text_input(
        "Filter by name or city",
        placeholder="e.g. Korle Bu, Tamale, Baptist..."
    )

    dir_df = filtered_df.copy()
    dir_df['address_city'] = dir_df['address_city'].fillna('').astype(str)

    if dir_search:
        mask = (
            dir_df['name'].str.lower().str.contains(dir_search.lower(), na=False) |
            dir_df['address_city'].str.lower().str.contains(dir_search.lower(), na=False)
        )
        dir_df = dir_df[mask]
        st.caption(f"Found {len(dir_df)} matching facilities")

    dcols = ['name', 'region_clean', 'address_city', 'facilityTypeId']
    for c in ['capability_text', 'procedure_text', 'specialties_text', 'confidence']:
        if c in dir_df.columns:
            dcols.append(c)

    st.dataframe(
        dir_df[dcols].rename(columns={
            'name': 'Hospital', 'region_clean': 'Region',
            'address_city': 'City', 'facilityTypeId': 'Type',
            'capability_text': 'Capabilities',
            'procedure_text': 'Procedures',
            'specialties_text': 'Specialties',
            'confidence': 'Confidence',
        }),
        use_container_width=True, hide_index=True, height=520
    )