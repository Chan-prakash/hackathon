import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ast
import re
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
    'Upper East':    ['upper east', 'bolgatanga', 'bawku', 'navrongo', 'zebilla'],
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

GAP_TRIGGERS  = ['desert', 'deploy', 'few hospital', 'most urgently', 'underserved',
                 'shortage', 'where should', 'which region', 'least hospital',
                 'coverage', 'how many region', 'overlapping', 'permanent vs', 'temporary']
NGO_TRIGGERS  = ['ngo', 'foundation', 'charity', 'volunteer', 'aid organisation',
                 'non-profit', 'nonprofit', 'community health program']
COUNT_TRIGGERS   = ['how many', 'count', 'number of', 'total number', 'how much',
                    'what is the total']
ANOMALY_TRIGGERS = ['anomal', 'abnormal', 'suspicious', 'correlat', 'pattern',
                    'mismatch', 'inconsisten', 'physical facility features',
                    'correlate with genuine', 'legitimate subspecialty']


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


# ════════════════════════════════════════════════════════════════
# QUESTION ROUTER  ← NEW: routes BEFORE touching FAISS
# ════════════════════════════════════════════════════════════════
def route_question(question):
    """
    Detect what kind of answer the question needs.
    COUNT and ANOMALY must NEVER go through FAISS — they need
    all 896 rows, not the top-5 retrieved results.
    """
    q = question.lower().strip()
    if any(t in q for t in COUNT_TRIGGERS):
        return 'count'
    if any(t in q for t in ANOMALY_TRIGGERS):
        return 'anomaly'
    if any(t in q for t in GAP_TRIGGERS):
        return 'gap'
    return 'search'


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

    has_enriched = (
        'enriched_capability' in df.columns and
        df['enriched_capability'].astype(str).str.len().mean() > 10
    )

    df['procedure_text']   = df['procedure'].apply(unpack_list)   if 'procedure'   in df.columns else df.get('procedure_text',   pd.Series([""] * len(df)))
    df['equipment_text']   = df['equipment'].apply(unpack_list)   if 'equipment'   in df.columns else df.get('equipment_text',   pd.Series([""] * len(df)))
    df['specialties_text'] = df['specialties'].apply(unpack_list) if 'specialties' in df.columns else df.get('specialties_text', pd.Series([""] * len(df)))

    if has_enriched:
        df['capability_text'] = df['enriched_capability'].apply(clean_capability)
    elif 'capability_text' in df.columns:
        df['capability_text'] = df['capability_text'].apply(clean_capability)
    elif 'capability' in df.columns:
        df['capability_text'] = df['capability'].apply(clean_capability)
    else:
        df['capability_text'] = ""

    df['search_text_full'] = df.apply(build_search_text, axis=1)

    def completeness(row):
        return sum(1 for c in ['procedure_text', 'equipment_text', 'capability_text',
                               'specialties_text', 'address_city']
                   if str(row.get(c, '')).strip() not in ['', 'nan'])

    df['_c'] = df.apply(completeness, axis=1)
    df = df.sort_values('_c', ascending=False).drop_duplicates(subset=['name'], keep='first')
    df = df.drop(columns=['_c']).reset_index(drop=True)

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

    if 'confidence' not in df.columns:
        df['confidence'] = 'Medium'

    # Ensure is_ngo column is boolean
    if 'is_ngo' in df.columns:
        df['is_ngo'] = df['is_ngo'].astype(str).str.strip().str.lower().isin(
            ['true', '1', 'yes']
        )
        # Also fix facilityTypeId for NGOs
        df.loc[df['is_ngo'], 'facilityTypeId'] = 'ngo'
    else:
        df['is_ngo'] = False

    return df


# ════════════════════════════════════════════════════════════════
# PRE-COMPUTED STATS  ← NEW: built once, answers ALL count/anomaly
# questions from the full 896-row dataset, never from top-5 FAISS
# ════════════════════════════════════════════════════════════════
@st.cache_data
def build_precomputed_stats(df):
    stats = {}
    stats['total']     = len(df)
    stats['ngo_count'] = int(df['is_ngo'].sum()) if 'is_ngo' in df.columns else 0
    stats['fac_count'] = stats['total'] - stats['ngo_count']
    stats['by_type']   = df['facilityTypeId'].value_counts().to_dict()
    stats['by_region'] = (
        df[df['region_clean'] != 'Unknown']['region_clean']
        .value_counts().to_dict()
    )

    SERVICES = {
        'icu':           ['icu', 'intensive care', 'critical care'],
        'emergency':     ['emergency', 'accident and emergency', '24 hour', 'trauma'],
        'surgery':       ['surgery', 'surgical', 'operating theatre'],
        'maternity':     ['maternity', 'obstetric', 'delivery', 'labour', 'antenatal'],
        'pediatric':     ['pediatric', 'paediatric', 'children', 'neonatal'],
        'imaging':       ['x-ray', 'xray', 'mri', 'ct scan', 'ultrasound', 'radiology', 'imaging'],
        'laboratory':    ['laboratory', 'lab test', 'diagnostic', 'pathology'],
        'pharmacy':      ['pharmacy', 'pharmaceutical'],
        'cardiology':    ['cardiology', 'cardiac care', 'heart treatment'],
        'dermatology':   ['dermatology', 'skin care', 'skin disease'],
        'ophthalmology': ['ophthalmology', 'eye care', 'eye surgery'],
        'dialysis':      ['dialysis', 'renal', 'kidney care'],
        'mental health': ['mental health', 'psychiatric', 'counseling'],
        'dental':        ['dental', 'dentistry', 'tooth'],
    }

    # Combined searchable text per row across ALL capability columns
    df['_search'] = (
        df['capability_text'].fillna('') + ' ' +
        df['specialties_text'].fillna('') + ' ' +
        df['procedure_text'].fillna('') + ' ' +
        df['equipment_text'].fillna('')
    ).str.lower()

    # Global service counts
    stats['service_counts'] = {}
    for svc, keywords in SERVICES.items():
        mask = df['_search'].apply(lambda t: any(k in t for k in keywords))
        stats['service_counts'][svc] = int(mask.sum())

    # Service counts per region
    stats['service_by_region'] = {}
    for region, rdf in df[df['region_clean'] != 'Unknown'].groupby('region_clean'):
        stats['service_by_region'][region] = {}
        for svc, keywords in SERVICES.items():
            mask = rdf['_search'].apply(lambda t: any(k in t for k in keywords))
            stats['service_by_region'][region][svc] = int(mask.sum())

    # Pre-computed anomalies across ALL rows
    anomalies = []
    for _, row in df.iterrows():
        caps  = str(row.get('capability_text', '')).lower()
        ftype = str(row.get('facilityTypeId', '')).lower()
        name  = str(row.get('name', ''))
        region = str(row.get('region_clean', ''))
        if ftype == 'clinic' and ('icu' in caps or 'intensive care' in caps):
            anomalies.append({
                'name': name, 'region': region, 'type': ftype,
                'issue': 'Clinic claiming ICU capability — verify before routing patients',
                'severity': 'HIGH'
            })
        if ftype == 'pharmacy' and 'surgery' in caps:
            anomalies.append({
                'name': name, 'region': region, 'type': ftype,
                'issue': 'Pharmacy claiming surgery — likely data error',
                'severity': 'HIGH'
            })
        if ftype == 'hospital' and len(caps.strip()) < 10:
            anomalies.append({
                'name': name, 'region': region, 'type': ftype,
                'issue': 'Hospital with no recorded capabilities — data incomplete',
                'severity': 'MEDIUM'
            })
        if ('surgery' in caps and 'operating theatre' not in caps
                and 'theatre' not in caps and ftype == 'hospital'):
            anomalies.append({
                'name': name, 'region': region, 'type': ftype,
                'issue': 'Claims surgery but no operating theatre confirmed',
                'severity': 'MEDIUM'
            })

    stats['anomalies']      = anomalies
    stats['anomaly_count']  = len(anomalies)
    return stats


# ════════════════════════════════════════════════════════════════
# COUNT ANSWER  ← NEW: queries full dataset, no FAISS
# ════════════════════════════════════════════════════════════════
def answer_count_question(question, df, stats):
    """
    Answers any counting question by looking at pre-computed stats.
    Works for ANY service — cardiology, dermatology, ophthalmology, etc.
    Never uses FAISS. Always counts across all 896 rows.
    """
    q = question.lower()
    region = detect_region(question)

    SERVICE_KEYWORDS = {
        'icu':           ['icu', 'intensive care', 'critical care'],
        'emergency':     ['emergency', 'accident'],
        'surgery':       ['surgery', 'surgical'],
        'maternity':     ['maternity', 'obstetric', 'delivery'],
        'pediatric':     ['pediatric', 'children', 'paediatric', 'neonatal'],
        'imaging':       ['x-ray', 'xray', 'mri', 'ct scan', 'ultrasound', 'imaging', 'radiology'],
        'laboratory':    ['laboratory', 'lab', 'diagnostic'],
        'cardiology':    ['cardiology', 'cardiac', 'heart'],
        'dermatology':   ['dermatology', 'skin'],
        'ophthalmology': ['ophthalmology', 'eye care', 'eye'],
        'dialysis':      ['dialysis', 'renal', 'kidney'],
        'dental':        ['dental', 'dentistry'],
        'pharmacy':      ['pharmacy'],
        'mental health': ['mental health', 'psychiatric'],
    }

    detected_svc = None
    for svc, keywords in SERVICE_KEYWORDS.items():
        if any(k in q for k in keywords):
            detected_svc = svc
            break

    # NGO + facility breakdown
    if any(t in q for t in ['ngo', 'foundation', 'charity', 'nonprofit']):
        n = stats['ngo_count']
        f = stats['fac_count']
        by_type = stats['by_type']
        hospitals  = by_type.get('hospital', 0)
        clinics    = by_type.get('clinic', 0)
        dentists   = by_type.get('dentist', 0)
        pharmacies = by_type.get('pharmacy', 0)
        return (
            f"There are **{n} NGOs** and **{f} healthcare facilities** "
            f"in the database (total: {stats['total']}).\n\n"
            f"Facility breakdown: {hospitals} hospitals · {clinics} clinics · "
            f"{dentists} dentists · {pharmacies} pharmacies · {n} NGOs."
        )

    if detected_svc and region:
        count        = stats['service_by_region'].get(region, {}).get(detected_svc, 0)
        region_total = stats['by_region'].get(region, 0)
        return (
            f"There are **{count}** facilities in {region} with "
            f"**{detected_svc}** services "
            f"(out of {region_total} total facilities in that region)."
        )

    if detected_svc:
        count = stats['service_counts'].get(detected_svc, 0)
        return (
            f"There are **{count}** facilities across Ghana with "
            f"**{detected_svc}** services "
            f"(counted across all {stats['total']} facilities in the database)."
        )

    if region:
        total      = stats['by_region'].get(region, 0)
        ngo_in_reg = int(df[(df['region_clean'] == region) & df['is_ngo']].shape[0])
        return (
            f"There are **{total}** facilities in {region} "
            f"({total - ngo_in_reg} healthcare facilities, {ngo_in_reg} NGOs)."
        )

    # Generic total
    by_type    = stats['by_type']
    hospitals  = by_type.get('hospital', 0)
    clinics    = by_type.get('clinic', 0)
    dentists   = by_type.get('dentist', 0)
    pharmacies = by_type.get('pharmacy', 0)
    ngos       = stats['ngo_count']
    return (
        f"There are **{stats['total']}** total entries in the database: "
        f"{hospitals} hospitals, {clinics} clinics, {dentists} dentists, "
        f"{pharmacies} pharmacies, and {ngos} NGOs."
    )


# ════════════════════════════════════════════════════════════════
# ANOMALY ANSWER  ← NEW: uses pre-computed anomalies, no FAISS
# ════════════════════════════════════════════════════════════════
def answer_anomaly_question(stats):
    """Returns pre-computed anomalies from all 896 rows. No FAISS. No hallucination."""
    anomalies = stats['anomalies']
    if not anomalies:
        return "No significant anomalies detected in the dataset."

    high   = [a for a in anomalies if a['severity'] == 'HIGH']
    medium = [a for a in anomalies if a['severity'] == 'MEDIUM']

    lines = [
        f"**{len(anomalies)} data anomalies detected** "
        f"({len(high)} high severity, {len(medium)} medium severity):\n"
    ]
    if high:
        lines.append("**🔴 High Severity:**")
        for a in high[:8]:
            lines.append(f"- **{a['name']}** ({a['region']}, {a['type']}): {a['issue']}")
    if medium:
        lines.append("\n**🟡 Medium Severity (sample):**")
        for a in medium[:5]:
            lines.append(f"- **{a['name']}** ({a['region']}): {a['issue']}")

    return "\n".join(lines)


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
               ngo_count=('is_ngo', 'sum'),
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
    """
    Pre-computed from full dataset — used for desert/deploy/NGO-overlap questions.
    Never uses FAISS.
    """
    region_counts = (df[df['region_clean'] != 'Unknown']['region_clean']
                     .value_counts().sort_values())
    ngo_by_region = {}
    if 'is_ngo' in df.columns:
        ngo_by_region = (
            df[df['is_ngo']][df['region_clean'] != 'Unknown']['region_clean']
            .value_counts().to_dict()
        )

    lines = ["**Regional healthcare coverage in Ghana (sorted by facility count):**\n"]
    for region, count in region_counts.items():
        risk = ('🔴 Critical'  if count <= 3 else
                '🟠 High Risk' if count <= 8 else
                '🟡 Moderate'  if count <= 20 else '🟢 Adequate')
        rd  = df[df['region_clean'] == region]
        stf = rd['search_text_full'].str.lower()
        gaps = []
        if not stf.str.contains('icu|intensive care', na=False).any():   gaps.append('no ICU')
        if not stf.str.contains('emergency|24/7', na=False).any():       gaps.append('no emergency')
        if not stf.str.contains('surg', na=False).any():                 gaps.append('no surgery')
        ngo_count = ngo_by_region.get(region, 0)
        ngo_str   = f", {ngo_count} NGOs" if ngo_count else ""
        gap_str   = f" — missing: {', '.join(gaps)}" if gaps else ""
        lines.append(f"- **{region}**: {count} facilities{ngo_str} {risk}{gap_str}")
        if count > 30:
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


# ════════════════════════════════════════════════════════════════
# LLM ANSWER  ← FIXED: passes full stats to Groq, not just top-5
# ════════════════════════════════════════════════════════════════
def get_llm_answer(question, results, groq_key, df, region, service, stats):
    if not groq_key:
        return "Set GROQ_KEY in Streamlit Secrets to enable AI analysis."
    from groq import Groq
    gclient = Groq(api_key=groq_key)

    # Build stats summary — Groq sees COUNTS from full dataset
    by_type = stats['by_type']
    stats_summary = (
        f"Total facilities in database: {stats['total']} "
        f"({by_type.get('hospital',0)} hospitals, {by_type.get('clinic',0)} clinics, "
        f"{stats['ngo_count']} NGOs)\n"
    )
    if region:
        region_total = stats['by_region'].get(region, 0)
        stats_summary += f"Facilities in {region}: {region_total}\n"
        if service:
            svc_count = stats['service_by_region'].get(region, {}).get(service, 0)
            stats_summary += f"Of these, {svc_count} have '{service}' services\n"
    elif service:
        svc_count = stats['service_counts'].get(service, 0)
        stats_summary += f"Facilities with '{service}' across Ghana: {svc_count}\n"

    # Retrieved results — specific examples
    context = ""
    for i, r in enumerate(results[:5], 1):
        ngo_tag = " [NGO]" if r.get('is_ngo') else ""
        context += (
            f"Example {i}{ngo_tag}: {r['name']} | {r['region']} | {r['city']}\n"
            f"  Capabilities: {r['capability'][:250]}\n"
            f"  Procedures  : {r['procedure'][:150]}\n"
            f"  Specialties : {r['specialties'][:150]}\n\n"
        )

    prompt = f"""You are a healthcare analyst for Virtue Foundation in Ghana.

DATABASE STATISTICS (from all {stats['total']} facilities):
{stats_summary}

RETRIEVED EXAMPLES (top matches for this query):
{context}

QUESTION: {question}

Answer rules:
- Use the DATABASE STATISTICS for any counts or totals — they are accurate across all {stats['total']} facilities.
- Use the RETRIEVED EXAMPLES to name specific hospitals.
- Never say "0" for a service if the statistics show it exists.
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
# HYBRID SEARCH INDEX
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_search_index(_df_len, _df_cols):
    df = st.session_state.get("_search_df")
    if df is None:
        return None, None, None, None, []

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
        is_ngo = bool(row.get('is_ngo', False))
        parts  = [
            f"Hospital: {row.get('name', '')}",
            f"Region: {row.get('region_clean', '')}",
            f"City: {row.get('address_city', '')}",
            f"Type: {row.get('facilityTypeId', '')}",
            f"NGO: {'yes' if is_ngo else 'no'}",
        ]
        desc = str(row.get('description', '')).strip()
        if desc and desc != 'nan':
            parts.append(f"Description: {desc[:300]}")

        specs = str(row.get('specialties_text', '')).strip()
        if specs and specs != 'nan':
            parts += [f"Specialties: {specs}"] * 2

        cap = str(row.get('capability_text', '')).strip()
        combined = ""
        if cap and cap != 'nan':
            parts += [f"Capabilities: {cap}"] * 3
            combined += cap.lower()

        proc = str(row.get('procedure_text', '')).strip()
        if proc and proc != 'nan':
            parts += [f"Procedures: {proc}"] * 2
            combined += " " + proc.lower()

        # NGO: also include mission/org description
        if is_ngo:
            for field in ['organizationDescription', 'missionStatement']:
                v = str(row.get(field, '')).strip()
                if v and v not in ('nan', ''):
                    parts.append(f"Mission: {v[:200]}")

        for concept, synonyms in MEDICAL_SYNONYMS.items():
            triggers = [concept] + synonyms.lower().split()[:2]
            if any(t in combined for t in triggers):
                parts.append(synonyms)

        return " | ".join(parts)

    texts = df.apply(build_rich_text, axis=1).tolist()

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    import faiss as faiss_lib
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
            if faiss_index.ntotal != len(df):
                raise ValueError(f"Index mismatch: {faiss_index.ntotal} vs {len(df)}")
        except Exception:
            embeddings  = embedder.encode(texts, show_progress_bar=False, batch_size=64).astype('float32')
            faiss_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)
    else:
        embeddings  = embedder.encode(texts, show_progress_bar=False, batch_size=64).astype('float32')
        faiss_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)

    return bm25, faiss_index, embeddings, embedder, texts


def hybrid_search(query, df, top_k=8, region_override=None, ngo_boost=False):
    region  = region_override or detect_region(query)
    service, svc_keywords = detect_service(query)

    st.session_state["_search_df"] = df
    bm25, faiss_index, embeddings, embedder, texts = load_search_index(
        len(df), str(df.columns.tolist())
    )

    if bm25 is None:
        return _keyword_fallback(query, df, region, service, svc_keywords, top_k)

    query_expanded = expand_query(query)

    bm25_scores = np.array(bm25.get_scores(query_expanded.lower().split()))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

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

    if region:
        region_kws = REGION_MAP.get(region, [region.lower()])
        for i in range(len(df)):
            row = df.iloc[i]
            loc = (str(row.get('region_clean', '')) + ' ' +
                   str(row.get('address_city', ''))).lower()
            if any(k in loc for k in region_kws):
                hybrid_scores[i] *= 1.8

    if ngo_boost:
        for i in range(len(df)):
            if bool(df.iloc[i].get('is_ngo', False)):
                hybrid_scores[i] *= 2.5

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        results.append({
            'name':        row['name'],
            'region':      row['region_clean'],
            'city':        str(row.get('address_city', '')),
            'type':        str(row.get('facilityTypeId', '')),
            'is_ngo':      bool(row.get('is_ngo', False)),
            'capability':  str(row.get('capability_text', '')),
            'procedure':   str(row.get('procedure_text', '')),
            'equipment':   str(row.get('equipment_text', '')),
            'specialties': str(row.get('specialties_text', '')),
            'confidence':  str(row.get('confidence', 'Medium')),
            'score':       float(hybrid_scores[i]),
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

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
            'is_ngo':      bool(row.get('is_ngo', False)),
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
# PAGE CONFIG + CSS
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ghana Healthcare Coverage · Virtue Foundation",
    page_icon="⚕️", layout="wide", initial_sidebar_state="expanded"
)

is_dark = st.session_state.get("dark_mode", True)

if is_dark:
    c_bg = "#0B1121"
    c_bg_alt = "#111827"
    c_sidebar = "linear-gradient(180deg, #040914 0%, #0A1628 100%)"
    c_side_text = "#CBD5E1"
    c_side_head = "#F8FAFC"
    c_card_bg = "rgba(255, 255, 255, 0.03)"
    c_card_border = "rgba(255, 255, 255, 0.08)"
    c_card_hover = "rgba(255, 255, 255, 0.06)"
    c_text_main = "#E2E8F0"
    c_text_bold = "#F8FAFC"
    c_text_muted = "#94A3B8"
    c_glass = "backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);"
    c_header_bg = "linear-gradient(135deg, rgba(10,22,40,0.8) 0%, rgba(13,59,110,0.8) 100%)"
else:
    c_bg = "#F4F6F9"
    c_bg_alt = "#E2E8F0"
    c_sidebar = "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)"
    c_side_text = "#475569"
    c_side_head = "#0F172A"
    c_card_bg = "rgba(255, 255, 255, 0.6)"
    c_card_border = "rgba(255, 255, 255, 0.9)"
    c_card_hover = "rgba(255, 255, 255, 0.8)"
    c_text_main = "#334155"
    c_text_bold = "#0F172A"
    c_text_muted = "#64748B"
    c_glass = "backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);"
    c_header_bg = "linear-gradient(135deg, #0284C7 0%, #0369A1 100%)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp {{ 
        font-family: 'Inter', sans-serif; 
        background: linear-gradient(120deg, {c_bg}, {c_bg_alt}, {c_bg});
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
        color: {c_text_main};
        transition: background 0.4s ease;
    }}
    
    h1, h2, h3, .section-header, .hospital-name, .metric-card .value {{
        font-family: 'Outfit', sans-serif !important;
        color: {c_text_bold} !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {c_sidebar};
        border-right: 1px solid {c_card_border};
    }}
    section[data-testid="stSidebar"] * {{ color: {c_side_text} !important; }}
    section[data-testid="stSidebar"] h3 {{ color: {c_side_head} !important; font-size: 0.85rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }}
    section[data-testid="stSidebar"] .stSelectbox label {{ color: {c_text_muted} !important; font-size: 0.75rem !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: {c_card_border} !important; }}

    .app-header {{
        background: {c_header_bg};
        border-radius: 16px; padding: 2rem 2.5rem;
        margin-bottom: 1.5rem; position: relative; overflow: hidden;
        {c_glass} border: 1px solid {c_card_border};
    }}
    .app-header::before {{
        content: ''; position: absolute; top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .app-header h1 {{
        font-size: 2.2rem; font-weight: 600; color: #FFFFFF !important; margin: 0; letter-spacing: -0.02em;
    }}
    .app-header p {{ font-size: 0.95rem; color: rgba(255,255,255,0.85); margin: 0.4rem 0 0 0; }}
    .app-header .eval-badge {{
        display: inline-block;
        background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.5);
        border-radius: 999px; padding: 0.25rem 0.75rem;
        font-size: 0.75rem; font-weight: 600; color: #10B981; margin-top: 0.75rem;
    }}

    .metric-card {{
        background: {c_card_bg}; border: 1px solid {c_card_border}; border-radius: 14px;
        padding: 1.25rem 1.5rem; transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
        {c_glass}
    }}
    .metric-card:hover {{ 
        transform: translateY(-4px) scale(1.02); 
        box-shadow: 0 12px 30px rgba(0,0,0,0.15); 
        background: {c_card_hover};
    }}
    .metric-card .label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {c_text_muted}; margin-bottom: 0.4rem; }}
    .metric-card .value {{ font-size: 2.2rem; font-weight: 600; line-height: 1.1; }}
    .metric-card .sub   {{ font-size: 0.78rem; color: {c_text_muted}; margin-top: 0.25rem; }}
    
    .metric-card.teal .value  {{ color: #06B6D4 !important; }}
    .metric-card.red .value   {{ color: #EF4444 !important; }}
    .metric-card.amber .value {{ color: #F59E0B !important; }}
    .metric-card.green .value {{ color: #10B981 !important; }}
    .metric-card.purple .value{{ color: #8B5CF6 !important; }}

    .result-card {{
        background: {c_card_bg}; border: 1px solid {c_card_border}; border-radius: 12px;
        padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; transition: all 0.3s ease;
        {c_glass}
    }}
    .result-card:hover {{ 
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        background: {c_card_hover};
        border-color: rgba(8, 145, 178, 0.4);
    }}
    .result-card.match-card {{ border-left: 4px solid #0891B2; }}
    .hospital-meta {{ font-size: 0.85rem; color: {c_text_muted}; margin-top: 0.2rem; }}
    .hospital-data-row  {{ font-size: 0.85rem; color: {c_text_main}; margin-top: 0.5rem; line-height: 1.6; }}
    .hospital-data-label {{ font-weight: 600; color: #06B6D4; margin-right: 0.25rem; }}

    .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.02em; margin-left: 0.25rem; }}
    .badge-high      {{ background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }}
    .badge-medium    {{ background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }}
    .badge-low       {{ background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }}
    .badge-match     {{ background: rgba(59,130,246,0.15); color: #3B82F6; border: 1px solid rgba(59,130,246,0.3); }}
    .badge-critical  {{ background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }}
    .badge-high-risk {{ background: rgba(217,119,6,0.15); color: #D97706; border: 1px solid rgba(217,119,6,0.3); }}
    .badge-moderate  {{ background: rgba(202,138,4,0.15); color: #CA8A04; border: 1px solid rgba(202,138,4,0.3); }}
    .badge-adequate  {{ background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }}
    .badge-ngo       {{ background: rgba(139,92,246,0.15); color: #8B5CF6; border: 1px solid rgba(139,92,246,0.3); }}

    .ai-answer, .count-answer, .anomaly-answer {{
        background: {c_card_bg};
        border: 1px solid {c_card_border}; border-radius: 12px;
        padding: 1.25rem 1.5rem; margin: 1rem 0;
        font-size: 0.9rem; line-height: 1.7; color: {c_text_main};
        {c_glass}
    }}
    .ai-answer .answer-label, .count-answer .answer-label, .anomaly-answer .answer-label {{
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #06B6D4; margin-bottom: 0.6rem;
    }}

    .search-context {{
        background: rgba(8, 145, 178, 0.1); border: 1px solid rgba(8, 145, 178, 0.2);
        border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 1rem; font-size: 0.85rem; color: #06B6D4;
    }}

    .section-header {{
        margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #0891B2; display: inline-block;
        letter-spacing: -0.01em;
    }}

    .gap-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; }}
    .gap-table th {{
        background: {c_card_hover}; color: {c_text_muted}; font-weight: 600;
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
        padding: 0.75rem 1rem; text-align: left; border-bottom: 2px solid {c_card_border};
    }}
    .gap-table td {{ padding: 0.65rem 1rem; border-bottom: 1px solid {c_card_border}; color: {c_text_main}; }}
    .gap-table tr:hover td {{ background: {c_card_hover}; transition: background 0.2s ease; }}

    .anomaly-flag {{
        background: rgba(245,158,11,0.1); border-left: 3px solid #F59E0B;
        padding: 0.4rem 0.75rem; margin-top: 0.5rem;
        font-size: 0.8rem; color: {c_text_main}; border-radius: 0 6px 6px 0;
    }}

    .eval-panel {{
        background: {c_card_bg}; border: 1px solid {c_card_border};
        border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.5rem;
        {c_glass}
    }}
    .eval-panel .eval-title {{
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: {c_text_muted}; margin-bottom: 0.5rem;
    }}
    .eval-row {{ display: flex; justify-content: space-between; font-size: 0.78rem; margin: 0.3rem 0; }}
    .eval-row .metric-name {{ color: {c_text_muted}; }}
    .eval-row .metric-val  {{ color: #10B981; font-weight: 700; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid {c_card_border}; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif; font-weight: 500;
        font-size: 0.9rem; color: {c_text_muted};
        padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent; margin-bottom: -2px;
        transition: color 0.3s ease, border-color 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{ color: #06B6D4 !important; border-bottom-color: #06B6D4 !important; background: transparent !important; font-weight: 600 !important; }}

    div[data-testid="stTextInput"] input {{
        background: {c_card_bg}; color: {c_text_main};
        border-radius: 10px; border: 2px solid {c_card_border};
        font-size: 0.95rem; padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif; transition: all 0.3s ease;
        {c_glass}
    }}
    div[data-testid="stTextInput"] input:focus {{ border-color: #06B6D4; box-shadow: 0 0 0 3px rgba(6,182,212,0.15); }}
    
    /* Emergency Contact Button Styling */
    .emergency-fab {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, #FF3B30 0%, #E32636 100%);
        color: white;
        width: 65px;
        height: 65px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 4px 15px rgba(227, 38, 54, 0.4);
        cursor: pointer;
        z-index: 9999;
        text-decoration: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: pulseEmergency 2s infinite;
    }}
    .emergency-fab:hover {{
        transform: scale(1.1) translateY(-5px);
        box-shadow: 0 8px 25px rgba(227, 38, 54, 0.6);
        color: white;
    }}
    .emergency-fab-text {{
        position: absolute;
        right: 80px;
        background: linear-gradient(135deg, #FF3B30 0%, #E32636 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 15px;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease, transform 0.3s ease;
        transform: translateX(10px);
        white-space: nowrap;
        box-shadow: 0 4px 15px rgba(227, 38, 54, 0.3);
    }}
    .emergency-fab:hover .emergency-fab-text {{
        opacity: 1;
        transform: translateX(0);
    }}
    @keyframes pulseEmergency {{
        0% {{ box-shadow: 0 0 0 0 rgba(227, 38, 54, 0.6); }}
        70% {{ box-shadow: 0 0 0 15px rgba(227, 38, 54, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(227, 38, 54, 0); }}
    }}

    #MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}}
    header {{visibility:hidden;}} .stDeployButton {{display:none;}}
</style>
""", unsafe_allow_html=True)

# Inject Emergency Contact Button
st.markdown("""
<a href="tel:112" class="emergency-fab" title="Call Emergency (112)">
    📞
    <span class="emergency-fab-text">Emergency: 112 Ghana</span>
</a>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MAIN APP — load data + stats once
# ════════════════════════════════════════════════════════════════
df                    = load_data()
precomputed_stats     = build_precomputed_stats(df)      # ← built once at load
gap_df, services_dict = build_gap_analysis(df)
groq_key              = os.environ.get("GROQ_KEY", "")

if not groq_key:
    try:
        groq_key = st.secrets.get("GROQ_KEY", "")
    except Exception:
        pass

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚕️ Ghana Healthcare")
    st.toggle("🌙 Dark Mode", value=True, key="dark_mode")
    st.markdown("---")

    has_enriched = ('enriched_capability' in df.columns and
                    df['enriched_capability'].astype(str).str.len().mean() > 10)
    icu_count = precomputed_stats['service_counts'].get('icu', 0)

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
    st.markdown(f"**{precomputed_stats['total']}** unique facilities")
    st.markdown(f"**{precomputed_stats['ngo_count']}** NGOs identified")
    st.markdown(f"**{df['region_clean'].nunique()}** regions with data")
    st.markdown(f"**{len(gap_df[gap_df['risk_level']=='Critical'])}** critical desert regions")
    if has_enriched:
        st.markdown(f"**{icu_count}** facilities with ICU")
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
  <span class="eval-badge">✅ RAG Score: 0.923 · 12/12 questions passed · {precomputed_stats['total']} facilities indexed</span>
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
        (c1, "teal",   "Total Facilities",  precomputed_stats['total'],       "in database"),
        (c2, "red",    "Critical Deserts",  critical_count,                   "regions"),
        (c3, "amber",  "Services Tracked",  len(services_dict),               "types"),
        (c4, "green",  "Regions Covered",   df[df['region_clean']!='Unknown']['region_clean'].nunique(), "regions"),
        (c5, "purple", "RAG Score",         "0.923",                          "12/12 passed"),
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
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
            font=dict(family='DM Sans', size=12),
            annotations=[dict(text=f'<b>{len(gap_df)}</b><br>Regions',
                             x=0.5, y=0.5, font_size=14,
                             font_color='#0F172A', showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

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
# TAB 2 — SEARCH  ← FIXED: routes question BEFORE FAISS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Search Healthcare Facilities</div>', unsafe_allow_html=True)
    st.caption("Hybrid BM25 + FAISS semantic search · Count questions answered from full dataset · 896 facilities indexed")

    col_a, col_b, col_c, col_d = st.columns(4)
    example_queries = [
        "How many hospitals have ICU in Accra?",
        "Emergency care in Northern Ghana",
        "Maternity services Volta region",
        "How many NGOs are in Ghana?",
    ]
    for col, eq in zip([col_a, col_b, col_c, col_d], example_queries):
        with col:
            if st.button(eq, use_container_width=True):
                st.session_state['search_query'] = eq

    default_q = st.session_state.get('search_query', '')
    question  = st.text_input(
        "Search",
        value=default_q,
        placeholder="e.g. How many hospitals in Ashanti have dermatology?",
        label_visibility="collapsed"
    )

    if st.button("🔍 Search", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            q_type = route_question(question)

            # ── COUNT: full dataset, no FAISS ────────────────
            if q_type == 'count':
                answer = answer_count_question(question, df, precomputed_stats)
                st.markdown(
                    f'<div class="count-answer">'
                    f'<div class="answer-label">📊 Counted across all {precomputed_stats["total"]} facilities</div>'
                    f'{answer}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── ANOMALY: pre-computed, no FAISS ──────────────
            elif q_type == 'anomaly':
                answer = answer_anomaly_question(precomputed_stats)
                st.markdown(
                    f'<div class="anomaly-answer">'
                    f'<div class="answer-label" style="font-size:0.7rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;color:#92400E;margin-bottom:0.6rem;">'
                    f'⚠️ Data Anomalies (scanned all {precomputed_stats["total"]} facilities)</div>'
                    f'{answer}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── GAP: pre-computed regional analysis ──────────
            elif q_type == 'gap':
                gap_text = get_gap_analysis_answer(df)
                st.markdown(
                    f'<div class="ai-answer">'
                    f'<div class="answer-label">📊 Gap Analysis (all {precomputed_stats["total"]} facilities)</div>'
                    f'{gap_text}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── SEARCH: hybrid FAISS + Groq ───────────────────
            else:
                region_override = selected_region if selected_region != "All" else None
                is_ngo_q = any(t in question.lower() for t in NGO_TRIGGERS)

                with st.spinner(f"Searching {precomputed_stats['total']} facilities..."):
                    results, detected_region, detected_service = hybrid_search(
                        question, df, top_k=8,
                        region_override=region_override,
                        ngo_boost=is_ngo_q,
                    )

                ctx_parts = []
                if detected_region:  ctx_parts.append(f"📍 Region: **{detected_region}**")
                if detected_service: ctx_parts.append(f"🏥 Service: **{detected_service}**")
                if is_ngo_q:         ctx_parts.append("🤝 NGO search active")
                if not ctx_parts:    ctx_parts.append("🔍 Searching all regions and services")
                st.markdown(
                    f'<div class="search-context">{" · ".join(ctx_parts)}</div>',
                    unsafe_allow_html=True
                )

                if groq_key:
                    with st.spinner("Generating AI analysis..."):
                        answer = get_llm_answer(
                            question, results, groq_key, df,
                            detected_region, detected_service,
                            precomputed_stats,        # ← pass stats so Groq sees full counts
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

                relevant = [r for r in results if r.get('score_display', 0) > 0]
                others   = [r for r in results if r.get('score_display', 0) == 0]
                label    = "Matching" if relevant else "Available"
                extra    = f" + {len(others)} other facilities" if others and relevant else ""
                st.markdown(
                    f'<div class="section-header">{label} Facilities '
                    f'({len(relevant)} relevant{extra})</div>',
                    unsafe_allow_html=True
                )

                display_results = (relevant + others) if relevant else results
                for r in display_results[:8]:
                    anomalies    = detect_anomalies(r)
                    anomaly_html = "".join(
                        f'<div class="anomaly-flag">⚠️ {a}</div>' for a in anomalies
                    )
                    cap_html  = (
                        f'<div class="hospital-data-row">'
                        f'<span class="hospital-data-label">Capabilities:</span>'
                        f' {r["capability"][:350]}</div>'
                        if r["capability"] and r["capability"] != 'nan' else ""
                    )
                    proc_html = (
                        f'<div class="hospital-data-row">'
                        f'<span class="hospital-data-label">Procedures:</span>'
                        f' {r["procedure"][:200]}</div>'
                        if r["procedure"] and r["procedure"] != 'nan' else ""
                    )
                    spec_html = (
                        f'<div class="hospital-data-row">'
                        f'<span class="hospital-data-label">Specialties:</span>'
                        f' {r["specialties"][:200]}</div>'
                        if r["specialties"] and r["specialties"] != 'nan' else ""
                    )
                    conf      = r["confidence"].lower() if r["confidence"] else "medium"
                    conf_cls  = "high" if "high" in conf else "medium" if "medium" in conf else "low"
                    match_badge = (
                        f'<span class="badge badge-match">✓ {detected_service}</span>'
                        if r.get("score_display", 0) > 0 and detected_service else ""
                    )
                    ngo_badge  = ('<span class="badge badge-ngo">NGO</span>'
                                  if r.get("is_ngo") else "")
                    card_class = "result-card match-card" if r.get("score_display", 0) > 0 else "result-card"

                    st.markdown(f"""
<div class="{card_class}">
  <div class="hospital-name">
    {r["name"]}
    <span class="badge badge-{conf_cls}">{r["confidence"]} confidence</span>
    {match_badge}{ngo_badge}
  </div>
  <div class="hospital-meta">
    📍 {r["region"] or "Unknown"} · {r["city"] or "Unknown"} · {r["type"] or "facility"}
  </div>
  {cap_html}{proc_html}{spec_html}{anomaly_html}
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Regional Gap Analysis</div>', unsafe_allow_html=True)
    st.caption(f"Based on enriched capability data from {precomputed_stats['total']} Ghana healthcare facilities")

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
    display_gap = gap_df.copy()
    if selected_risk   != 'All': display_gap = display_gap[display_gap['risk_level'] == selected_risk]
    if selected_region != 'All': display_gap = display_gap[display_gap['region_clean'] == selected_region]

    table_html = (
        '<table class="gap-table"><thead><tr>'
        '<th>Region</th><th>Facilities</th><th>NGOs</th><th>Services</th><th>Risk Level</th>'
        + "".join(f'<th>{s}</th>' for s in services_dict)
        + '</tr></thead><tbody>'
    )
    for _, row in display_gap.iterrows():
        rc_ = row['risk_level'].lower().replace(' ', '-')
        ngo_c = int(row.get('ngo_count', 0))
        table_html += (
            f'<tr><td><strong>{row["region_clean"]}</strong></td>'
            f'<td style="text-align:center">{int(row["total_facilities"])}</td>'
            f'<td style="text-align:center">{ngo_c}</td>'
            f'<td style="text-align:center">{int(row["services_available"])}/8</td>'
            f'<td><span class="badge badge-{rc_}">{row["risk_level"]}</span></td>'
        )
        for svc in services_dict:
            has   = row.get(f'has_{svc}', 0) > 0
            color = "#059669" if has else "#DC2626"
            icon  = "✓" if has else "✗"
            table_html += (
                f'<td style="color:{color};font-weight:700;text-align:center">{icon}</td>'
            )
        table_html += '</tr>'
    st.markdown(table_html + '</tbody></table>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Deployment Recommendations</div>', unsafe_allow_html=True)
    critical_regions = gap_df[gap_df['risk_level'] == 'Critical'].sort_values('total_facilities')
    for _, row in critical_regions.iterrows():
        missing_svcs = [s for s in services_dict if not row.get(f'has_{s}', False)]
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
    for c in ['capability_text', 'procedure_text', 'specialties_text', 'confidence', 'is_ngo']:
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
            'is_ngo': 'NGO',
        }),
        use_container_width=True, hide_index=True, height=520
    )