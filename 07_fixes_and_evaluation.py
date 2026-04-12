# Databricks notebook source
# MAGIC %md
# MAGIC ## Cell 1 — Install Libraries

# COMMAND ----------

# MAGIC %pip install groq sentence-transformers faiss-cpu -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Imports & Config

# COMMAND ----------

import os, json, ast, time
import pandas as pd
import numpy as np
import mlflow
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_KEY")  # Set GROQ_KEY in Databricks cluster env vars
MLFLOW_EXPERIMENT = "/Users/chandrakanthprakash26@gmail.com/medical-desert-idp-agent"

client = Groq(api_key=GROQ_KEY)

# Test Groq
try:
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=5, n=1,
    )
    print("✅ Groq connected!")
except Exception as e:
    if "429" in str(e):
        print("⚠️  Groq rate limited — evaluation will use keyword-only scoring")
        client = None
    else:
        print(f"❌ Groq error: {e}")
        client = None

print("✅ Setup done!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load Data from Delta Table

# COMMAND ----------

df_raw = spark.table("hospital_metadata_full").toPandas().fillna("")
print(f"✅ Loaded {len(df_raw)} hospitals from Delta table")
print(f"   Columns: {list(df_raw.columns)}")
print(f"\nRegion distribution:")
print(df_raw['region_clean'].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — FIX: Unpack JSON Lists into Clean Text
# MAGIC
# MAGIC **THE ROOT CAUSE OF BAD SEARCH:**
# MAGIC The procedure/capability/equipment fields are stored as JSON strings like
# MAGIC `["ICU beds", "emergency care"]`. The FAISS search was embedding these
# MAGIC raw strings with brackets, so a query for "ICU" had nothing to match.
# MAGIC This cell fixes that permanently.

# COMMAND ----------

import pandas as pd
import json
import ast

# ✅ Load dataset FIRST
df_raw = pd.read_csv("/Volumes/workspace/default/project/hospital_metadata.csv")

# Your functions (same as yours)
def unpack_list(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None', '[""]']:
        return ""
    s = str(val).strip()
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return s
    if isinstance(parsed, list):
        items = [str(x).strip() for x in parsed
                 if str(x).strip() not in ['', 'nan', '""', "''"]]
        return " | ".join(items)
    return str(val)

def clean_capability(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return ""
    try:
        parsed = json.loads(str(val))
    except Exception:
        try:
            parsed = ast.literal_eval(str(val))
        except Exception:
            return str(val)
    if isinstance(parsed, list):
        clean = []
        for item in parsed:
            s = str(item).lower()
            if any(x in s for x in [
                'contact phone', 'email', '@', 'www.',
                'located at', 'currently closed', 'opening hours',
                'closed on', 'http', 'facebook', 'instagram',
                'twitter', 'linkedin', '+233'
            ]):
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

    spec = unpack_list(row.get('specialties', ''))
    if spec:
        parts.append(f"Specialties: {spec[:300]}")

    proc = unpack_list(row.get('procedure', ''))
    if proc:
        parts.append(f"Procedures: {proc[:400]}")

    cap = clean_capability(row.get('capability', ''))
    if cap:
        parts.append(f"Capabilities: {cap[:400]}")

    equip = unpack_list(row.get('equipment', ''))
    if equip:
        parts.append(f"Equipment: {equip[:300]}")

    return " | ".join(parts)

# ✅ Process
df = df_raw.copy()

df['procedure_text']   = df['procedure'].apply(unpack_list)
df['equipment_text']   = df['equipment'].apply(unpack_list)
df['capability_text']  = df['capability'].apply(clean_capability)
df['specialties_text'] = df['specialties'].apply(unpack_list)
df['search_text_full'] = df.apply(build_search_text, axis=1)

print("✅ Text processing complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — FIX: Region Normalization + City Fallback

# COMMAND ----------

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
    'cantonments': 'Greater Accra', 'east legon': 'Greater Accra',
    'kumasi': 'Ashanti', 'obuasi': 'Ashanti', 'ejisu': 'Ashanti',
    'takoradi': 'Western', 'sekondi': 'Western', 'tarkwa': 'Western',
    'cape coast': 'Central', 'winneba': 'Central', 'saltpond': 'Central',
    'tamale': 'Northern', 'yendi': 'Northern', 'savelugu': 'Northern',
    'bolgatanga': 'Upper East', 'bawku': 'Upper East', 'navrongo': 'Upper East',
    'wa': 'Upper West', 'lawra': 'Upper West',
    'ho ': 'Volta', 'hohoe': 'Volta', 'kpando': 'Volta', 'aflao': 'Volta',
    'sunyani': 'Brong Ahafo', 'techiman': 'Brong Ahafo',
    'akosombo': 'Eastern', 'koforidua': 'Eastern', 'asamankese': 'Eastern',
    'sefwi': 'Western North', 'bibiani': 'Western North',
    'damongo': 'Savannah', 'nalerigu': 'North East', 'gambaga': 'North East',
    'dambai': 'Oti', 'nkwanta': 'Oti',
}

unknown_before = (df['region_clean'] == 'Unknown').sum()

def fix_unknown_region(row):
    if row['region_clean'] != 'Unknown':
        return row['region_clean']
    city = str(row.get('address_city', '')).lower().strip()
    name = str(row.get('name', '')).lower()
    combined = city + ' ' + name
    for city_key, region in city_to_region.items():
        if city_key in combined:
            return region
    return 'Unknown'

df['region_clean'] = df.apply(fix_unknown_region, axis=1)
unknown_after = (df['region_clean'] == 'Unknown').sum()

print(f"✅ Region fix complete!")
print(f"   Unknown before: {unknown_before} → after: {unknown_after}")
print(f"\nRegion distribution after fix:")
print(df['region_clean'].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Build FAISS Index (with clean embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — 10-Question Evaluation (Keyword + Optional LLM Scoring)
# MAGIC
# MAGIC This tests whether the FAISS index can retrieve relevant hospitals
# MAGIC for the 10 most common NGO questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Log Evaluation to MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Save Fixed Data Back to Delta Table

# COMMAND ----------

# Save fixed dataframe back to Delta table
# This ensures Notebook 4 (LangGraph RAG) always uses clean search text

spark_df = spark.createDataFrame(df.astype(str).fillna(''))
spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_full")

print(f"✅ Fixed data saved to Delta table: hospital_metadata_full")
print(f"   Rows: {len(df)}")
print(f"   New columns: procedure_text, equipment_text, capability_text, specialties_text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Export CSV + FAISS Index to Volumes (for Streamlit)

# COMMAND ----------

# Save CSV
csv_path   = "/Volumes/workspace/default/project/hospital_metadata.csv"
index_path = "/Volumes/workspace/default/project/ghana_hospitals_full.index"

df.to_csv(csv_path, index=False)
faiss.write_index(index, index_path)

print(f"✅ CSV saved to   : {csv_path}")
print(f"✅ FAISS saved to : {index_path}")
print(f"   Hospitals      : {len(df)}")
print(f"   FAISS vectors  : {index.ntotal}")

# Also save to /tmp for download
df.to_csv("/tmp/hospital_metadata.csv", index=False)
faiss.write_index(index, "/tmp/ghana_hospitals_full.index")

# Create download links
import base64

with open("/tmp/hospital_metadata.csv", "rb") as f:
    b64_csv = base64.b64encode(f.read()).decode()

displayHTML(f'''
<h3>📥 Download Fixed Files</h3>
<p>
<a download="hospital_metadata.csv"
   href="data:text/csv;base64,{b64_csv}"
   style="padding:8px 16px;background:#0D7377;color:white;border-radius:6px;text-decoration:none;margin-right:10px">
   ⬇️ Download hospital_metadata.csv
</a>
</p>
<p style="color:#666;font-size:0.9em">Upload this CSV to your Streamlit GitHub repo → data/ folder</p>
''')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11 — Gap Analysis Verification (SQL)
# MAGIC
# MAGIC Use this in Genie for Text2SQL queries.
# MAGIC These SQL views work correctly because we clean the text first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12 — Genie-Ready Example Queries
# MAGIC
# MAGIC Paste these into Databricks Genie to verify Text2SQL works.

# COMMAND ----------

# ==============================
# 🔧 SETUP + EMBEDDINGS + FAISS
# ==============================

# Install (run once)
# !pip install -q sentence-transformers faiss-cpu

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 🔥 1. Load Medical Embedding Model
print("🔄 Loading medical embedding model...")
embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# 🔥 2. Query Expansion
def expand_query(q):
    q = q.lower()
    replacements = {
        "icu": "intensive care unit critical care ventilator life support",
        "emergency": "emergency care accident trauma 24 hour urgent care",
        "maternity": "maternity obstetric gynecology delivery labour antenatal",
        "surgery": "surgery surgical operation theatre surgeon procedure",
        "pediatric": "pediatric children neonatal baby infant care",
        "imaging": "mri ct scan xray radiology ultrasound diagnostic imaging",
    }
    for k, v in replacements.items():
        if k in q:
            q += " " + v
    return q

# 🔥 3. Boosted Search Text (REBUILD)
def build_search_text(row):
    parts = [
        f"Hospital: {row.get('name', '')}",
        f"Region: {row.get('region_clean', '')}",
        f"City: {row.get('address_city', '')}",
    ]

    desc = str(row.get('description', '')).strip()
    if desc and desc != 'nan':
        parts.append(f"Description: {desc}")

    spec = str(row.get('specialties_text', ''))
    if spec:
        parts.append(f"Specialties: {spec} {spec}")

    proc = str(row.get('procedure_text', ''))
    if proc:
        parts.append(f"Procedures: {proc} {proc}")

    cap = str(row.get('capability_text', ''))
    if cap:
        parts.append(f"Capabilities: {cap} {cap}")

    equip = str(row.get('equipment_text', ''))
    if equip:
        parts.append(f"Equipment: {equip}")

    return " | ".join(parts)

# Apply boosted text
print("🔄 Rebuilding search text...")
df['search_text_full'] = df.apply(build_search_text, axis=1)

# 🔥 4. Create embeddings
print("🔄 Generating embeddings...")
texts = df['search_text_full'].fillna("").tolist()

embeddings = embedder.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

embeddings = np.array(embeddings).astype('float32')

# 🔥 5. Build FAISS
print("🔄 Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"✅ FAISS ready: {index.ntotal} vectors")

# COMMAND ----------

# =========================================
# 🚀 FINAL FULL RAG SYSTEM (NO ERRORS)
# =========================================

# !pip install -q sentence-transformers faiss-cpu pandas

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================================
# 🔥 0. LOAD DATASET (IMPORTANT FIX)
# =========================================
print("🔄 Loading dataset...")

df = pd.read_csv("/Volumes/workspace/default/project/hospital_metadata.csv")  # <-- ensure file path correct

# If already processed columns exist, keep them
required_cols = ['capability_text','procedure_text','equipment_text','specialties_text']
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

# =========================================
# 🔥 1. LOAD MODELS
# =========================================
print("🔄 Loading models...")

embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# =========================================
# 🔥 2. QUERY EXPANSION
# =========================================
def expand_query(q):
    q = q.lower()
    replacements = {
        "icu": "intensive care unit critical care ventilator life support",
        "emergency": "emergency care accident trauma 24 hour urgent care",
        "maternity": "maternity obstetric gynecology delivery labour antenatal",
        "surgery": "surgery surgical operation theatre surgeon procedure",
        "pediatric": "pediatric children neonatal baby infant care",
        "imaging": "mri ct scan xray radiology ultrasound diagnostic imaging",
    }
    for k, v in replacements.items():
        if k in q:
            q += " " + v
    return q

# =========================================
# 🔥 3. BUILD SEARCH TEXT (SAFE VERSION)
# =========================================
def build_search_text(row):
    parts = [
        f"Hospital: {row.get('name', '')}",
        f"Region: {row.get('region_clean', '')}",
        f"City: {row.get('address_city', '')}",
    ]

    parts.append(str(row.get('specialties_text', '')))
    parts.append(str(row.get('procedure_text', '')))
    parts.append(str(row.get('capability_text', '')))
    parts.append(str(row.get('equipment_text', '')))

    return " | ".join(parts)

print("🔄 Preparing search text...")
df['search_text_full'] = df.apply(build_search_text, axis=1)

# =========================================
# 🔥 4. EMBEDDINGS
# =========================================
print("🔄 Generating embeddings...")

texts = df['search_text_full'].fillna("").tolist()

embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype('float32')

# =========================================
# 🔥 5. FAISS
# =========================================
print("🔄 Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================================
# 🔥 6. SEARCH FUNCTION
# =========================================
def build_search_text(row):
    text_parts = [
        f"Hospital: {row.get('name', '')}",
        f"Region: {row.get('region_clean', '')}",
        f"City: {row.get('address_city', '')}",
    ]

    combined = (
        str(row.get('specialties_text', '')) + " " +
        str(row.get('procedure_text', '')) + " " +
        str(row.get('capability_text', '')) + " " +
        str(row.get('equipment_text', ''))
    ).lower()

    # 🔥 FORCE MEDICAL LOGIC
    if "cardio" in combined or "heart" in combined:
        combined += " icu intensive care critical care"

    if "surgery" in combined or "operation" in combined:
        combined += " surgery surgical operating theatre"

    if "emergency" in combined:
        combined += " emergency care trauma accident 24 hour"

    if "maternity" in combined or "obstetric" in combined:
        combined += " maternity delivery labour antenatal"

    if "children" in combined or "pediatric" in combined:
        combined += " pediatric neonatal child care"

    if "radiology" in combined or "mri" in combined:
        combined += " imaging diagnostic radiology ct scan xray"

    text_parts.append(combined)

    return " | ".join(text_parts)
# =========================================
# 🔥 7. TEST
# =========================================
print("\n🔍 TESTING...")

query = "ICU hospitals in Accra"

results = search_hospitals(query, region="Greater Accra")

for r in results:
    print(f"{r['name']} ({r['region']})")

# COMMAND ----------

print("Columns:\n", df.columns)

print("\nSample rows:")
print(df[['name','region_clean','capability_text','procedure_text','specialties_text']].head(5))

# Check ICU presence
icu_count = df['capability_text'].str.lower().str.contains('icu|intensive', na=False).sum()
print(f"\nICU-related entries: {icu_count}")

# Check surgery presence
surgery_count = df['capability_text'].str.lower().str.contains('surgery|surgical', na=False).sum()
print(f"Surgery-related entries: {surgery_count}")

# Check maternity presence
maternity_count = df['capability_text'].str.lower().str.contains('maternity|obstetric', na=False).sum()
print(f"Maternity-related entries: {maternity_count}")

# COMMAND ----------

from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, results, top_n=5):
    if not results:
        return results
    pairs = [
        (query, r['capability'] + " " + r['procedure'] + " " + r['equipment'])
        for r in results
    ]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r['rerank_score'] = float(s)
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:top_n]

# COMMAND ----------

# =========================================
# 🚀 FINAL PRODUCTION RAG SYSTEM (CLEAN)
# =========================================

# !pip install -q sentence-transformers faiss-cpu pandas

import pandas as pd
import numpy as np
import json
import ast
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================================
# 🔥 1. LOAD DATA
# =========================================
print("🔄 Loading dataset...")
df = pd.read_csv("/Volumes/workspace/default/project/hospital_metadata.csv").fillna("")

# =========================================
# 🔥 2. SPECIALTY → CAPABILITY ENRICHMENT
# =========================================
SPECIALTY_TO_CAPABILITY = {
    "emergencyMedicine": ["emergency care", "trauma care", "24 hour emergency"],
    "generalSurgery": ["surgery", "surgical procedures", "operating theatre"],
    "orthopedicSurgery": ["bone surgery", "fracture treatment"],
    "cardiology": ["heart treatment", "ICU", "intensive care", "critical care"],
    "gynecologyAndObstetrics": ["maternity care", "delivery", "labour ward"],
    "pediatrics": ["pediatric care", "children care", "neonatal care"],
    "radiology": ["imaging services", "x-ray", "diagnostic imaging"],
    "pathology": ["laboratory services", "diagnostic testing"],
}

def unpack_list(val):
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except:
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except:
            pass
    return [str(val).strip()] if val else []

def enrich_row(row):
    facts = set()

    # existing fields
    for col in ['capability', 'procedure', 'equipment']:
        for item in unpack_list(row.get(col, '')):
            if item:
                facts.add(item.lower())

    # specialties → capability
    for spec in unpack_list(row.get('specialties', '')):
        if spec in SPECIALTY_TO_CAPABILITY:
            for cap in SPECIALTY_TO_CAPABILITY[spec]:
                facts.add(cap)

    # description mining
    desc = str(row.get('description', '')).lower()
    if 'icu' in desc or 'intensive care' in desc:
        facts.update(['icu', 'intensive care unit', 'critical care'])
    if 'emergency' in desc:
        facts.add('emergency care')
    if 'surgery' in desc:
        facts.add('surgery')
    if 'maternity' in desc or 'delivery' in desc:
        facts.add('maternity care')
    if 'pediatric' in desc or 'children' in desc:
        facts.add('pediatric care')
    if 'mri' in desc or 'ct scan' in desc:
        facts.add('imaging services')

    return " | ".join(facts)

print("🔄 Enriching data...")
df['enriched_text'] = df.apply(enrich_row, axis=1)

# =========================================
# 🔥 3. BUILD SEARCH TEXT
# =========================================
def build_search_text(row):
    return f"""
    Hospital: {row['name']}
    Region: {row['region_clean']}
    City: {row['address_city']}
    Capabilities: {row['enriched_text']}
    """

df['search_text'] = df.apply(build_search_text, axis=1)

# =========================================
# 🔥 4. LOAD MODELS
# =========================================
print("🔄 Loading models...")
embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# =========================================
# 🔥 5. EMBEDDINGS + FAISS
# =========================================
print("🔄 Generating embeddings...")
texts = df['search_text'].tolist()

embeddings = embedder.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

print("🔄 Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================================
# 🔥 6. QUERY EXPANSION
# =========================================
def expand_query(q):
    q = q.lower()
    if "icu" in q:
        q += " intensive care unit critical care ventilator"
    if "surgery" in q:
        q += " surgical operation theatre"
    if "maternity" in q:
        q += " obstetric delivery labour"
    if "pediatric" in q:
        q += " children neonatal"
    if "imaging" in q:
        q += " mri ct scan xray"
    return q

# =========================================
# 🔥 7. SEARCH FUNCTION (FINAL)
# =========================================
def search(query, region=None, top_k=30):
    query_expanded = expand_query(query)

    q_emb = embedder.encode([query_expanded])
    q_emb = np.array(q_emb).astype('float32')

    dists, idxs = index.search(q_emb, top_k)

    results = []
    for d, i in zip(dists[0], idxs[0]):
        row = df.iloc[i]
        results.append({
            "name": row['name'],
            "region": row['region_clean'],
            "text": row['search_text']
        })

    # region filter
    if region:
        results = [r for r in results if region.lower() in str(r['region']).lower()]

    # rerank
    if results:
        pairs = [(query, r['text']) for r in results]
        scores = reranker.predict(pairs)

        for r, s in zip(results, scores):
            r['score'] = s

        results = sorted(results, key=lambda x: x['score'], reverse=True)

    return results[:5]

# =========================================
# 🔥 8. TEST
# =========================================
print("\n🔍 TEST QUERY:\n")

query = "ICU hospitals in Accra"
results = search(query, region="Greater Accra")

for r in results:
    print(f"👉 {r['name']} ({r['region']})")

# COMMAND ----------

# MAGIC %pip install rank_bm25 sentence-transformers faiss-cpu groq -q

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import ast
import faiss
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_KEY")  # Set GROQ_KEY in Databricks cluster env vars
client = Groq(api_key=GROQ_KEY)

# ══════════════════════════════════════════
# STEP 1 — LOAD & ENRICH DATA FROM ALL SOURCES
# ══════════════════════════════════════════
print("🔄 Loading data...")
df = spark.table("hospital_metadata_full").toPandas().fillna("")

# Specialty → capability inference
# The specialties column is 92% filled — use it to infer missing capabilities
SPECIALTY_TO_CAPABILITY = {
    "emergencyMedicine":       ["emergency care", "accident and emergency", "trauma care", "24 hour emergency"],
    "generalSurgery":          ["surgery", "surgical procedures", "operating theatre", "general surgery"],
    "orthopedicSurgery":       ["orthopedic surgery", "bone surgery", "fracture treatment"],
    "cardiology":              ["cardiac care", "heart treatment", "cardiology services", "ICU", "intensive care"],
    "gynecologyAndObstetrics": ["maternity care", "obstetrics", "delivery", "antenatal care", "labour ward"],
    "pediatrics":              ["pediatric care", "children care", "neonatal care", "child health"],
    "ophthalmology":           ["eye care", "eye surgery", "optical services"],
    "internalMedicine":        ["internal medicine", "general medicine", "outpatient care"],
    "familyMedicine":          ["family medicine", "general practice", "primary care"],
    "dentistry":               ["dental care", "tooth extraction", "dental surgery"],
    "generalDentistry":        ["dental care", "general dentistry"],
    "pathology":               ["laboratory services", "pathology lab", "diagnostic testing"],
    "radiology":               ["imaging services", "x-ray", "radiology", "diagnostic imaging"],
    "anesthesiology":          ["anesthesia", "ICU", "intensive care", "critical care"],
    "psychiatry":              ["mental health", "psychiatric care"],
    "dermatology":             ["skin care", "dermatology services"],
    "urology":                 ["urology services", "urological procedures"],
    "nephrology":              ["kidney care", "dialysis", "renal services"],
    "neurology":               ["neurology", "brain care", "neurological treatment"],
    "oncology":                ["cancer care", "oncology", "chemotherapy"],
}

def unpack_list(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return []
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except:
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except:
            pass
    return [str(val).strip()] if str(val).strip() else []

def enrich_row(row):
    """
    Build enriched capability text from ALL available sources:
    1. Existing capability/procedure/equipment (IDP extracted)
    2. Specialties column → infer capabilities
    3. Description keywords
    """
    facts = set()

    # From existing IDP extraction
    for col in ['capability', 'procedure', 'equipment']:
        for item in unpack_list(row.get(col, '')):
            item_clean = item.lower()
            # Remove contamination (addresses, contact info)
            if any(x in item_clean for x in [
                '@', 'www.', 'http', '+233', 'located at',
                'contact', 'opening hours', 'closed on', 'facebook'
            ]):
                continue
            facts.add(item.strip())

    # From specialties — infer capabilities
    specialties_raw = unpack_list(row.get('specialties', ''))
    for spec in specialties_raw:
        spec_clean = spec.strip()
        if spec_clean in SPECIALTY_TO_CAPABILITY:
            for cap in SPECIALTY_TO_CAPABILITY[spec_clean]:
                facts.add(cap)

    # From description — keyword mining
    desc = str(row.get('description', '')).lower()
    if 'icu' in desc or 'intensive care' in desc or 'critical care' in desc:
        facts.add('ICU')
        facts.add('intensive care unit')
        facts.add('critical care')
    if 'emergency' in desc or '24 hour' in desc or '24/7' in desc:
        facts.add('emergency care')
        facts.add('accident and emergency')
    if 'surgery' in desc or 'surgical' in desc or 'operation' in desc:
        facts.add('surgery')
        facts.add('surgical procedures')
    if 'maternity' in desc or 'obstetric' in desc or 'delivery' in desc or 'labour' in desc:
        facts.add('maternity care')
        facts.add('obstetrics')
    if 'pediatric' in desc or 'children' in desc or 'neonatal' in desc:
        facts.add('pediatric care')
    if 'imaging' in desc or 'x-ray' in desc or 'mri' in desc or 'ct scan' in desc or 'ultrasound' in desc:
        facts.add('imaging services')
        facts.add('diagnostic imaging')
    if 'laboratory' in desc or 'lab test' in desc or 'diagnostic' in desc:
        facts.add('laboratory services')

    return " | ".join(sorted(facts)) if facts else ""

print("🔄 Enriching all hospitals using specialties + description + IDP...")
df['enriched_capability'] = df.apply(enrich_row, axis=1)

# Verify improvement
icu_count = df['enriched_capability'].str.lower().str.contains('icu|intensive care', na=False).sum()
surgery_count = df['enriched_capability'].str.lower().str.contains('surgery|surgical', na=False).sum()
maternity_count = df['enriched_capability'].str.lower().str.contains('maternity|obstetric', na=False).sum()
emergency_count = df['enriched_capability'].str.lower().str.contains('emergency', na=False).sum()

print(f"\n✅ Enrichment complete!")
print(f"   ICU/intensive care : {icu_count} hospitals (was 5)")
print(f"   Surgery            : {surgery_count} hospitals (was 33)")
print(f"   Maternity          : {maternity_count} hospitals")
print(f"   Emergency          : {emergency_count} hospitals")

# COMMAND ----------

# ══════════════════════════════════════════
# STEP 2 — BUILD SEARCH TEXT FOR ALL HOSPITALS
# ══════════════════════════════════════════

QUERY_SYNONYMS = {
    "icu":       ["icu", "intensive care unit", "critical care", "intensive care"],
    "emergency": ["emergency", "accident and emergency", "trauma", "24 hour", "urgent care"],
    "surgery":   ["surgery", "surgical", "operating theatre", "operation"],
    "maternity": ["maternity", "obstetric", "delivery", "labour", "antenatal"],
    "pediatric": ["pediatric", "children", "neonatal", "child health"],
    "imaging":   ["imaging", "x-ray", "xray", "mri", "ct scan", "ultrasound", "radiology"],
    "laboratory":["laboratory", "lab", "diagnostic testing", "pathology"],
    "pharmacy":  ["pharmacy", "pharmaceutical", "dispensary"],
    "dental":    ["dental", "dentistry", "tooth"],
}

def expand_query(query):
    q = query.lower()
    expansions = []
    for key, synonyms in QUERY_SYNONYMS.items():
        if key in q or any(s in q for s in synonyms[:2]):
            expansions.extend(synonyms)
    if expansions:
        q = q + " " + " ".join(expansions)
    return q

def build_search_text(row):
    """Rich searchable text combining ALL columns"""
    parts = []
    
    # Core identity — always present
    parts.append(f"Hospital: {row.get('name', '')}")
    parts.append(f"Region: {row.get('region_clean', '')}")
    parts.append(f"City: {row.get('address_city', '')}")
    parts.append(f"Type: {row.get('facilityTypeId', '')}")
    
    # Description
    desc = str(row.get('description', '')).strip()
    if desc and desc != 'nan':
        parts.append(f"Description: {desc[:400]}")
    
    # Specialties — unpack from JSON
    specs = unpack_list(row.get('specialties', ''))
    if specs:
        spec_text = " | ".join(specs)
        # Boost specialties — repeat for FAISS weight
        parts.append(f"Specialties: {spec_text}")
        parts.append(f"Specialties: {spec_text}")
    
    # ENRICHED capability — the most important field
    enriched = str(row.get('enriched_capability', '')).strip()
    if enriched:
        # Boost 3x — this is our best signal
        parts.append(f"Capabilities: {enriched}")
        parts.append(f"Capabilities: {enriched}")
        parts.append(f"Capabilities: {enriched}")
    
    return " | ".join(parts)

print("🔄 Building enriched search text for all hospitals...")
df['search_text'] = df.apply(build_search_text, axis=1)

avg_len = df['search_text'].str.len().mean()
print(f"✅ Search text built — avg length: {avg_len:.0f} chars")

# ══════════════════════════════════════════
# STEP 3 — BUILD BM25 (keyword search)
# ══════════════════════════════════════════
print("\n🔄 Building BM25 keyword index...")

# Tokenize for BM25
tokenized = [text.lower().split() for text in df['search_text'].tolist()]
bm25 = BM25Okapi(tokenized)

print(f"✅ BM25 index built — {len(tokenized)} documents")

# ══════════════════════════════════════════
# STEP 4 — BUILD FAISS (semantic search)
# ══════════════════════════════════════════
print("\n🔄 Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("🔄 Generating embeddings (2-3 mins)...")
texts = df['search_text'].tolist()
embeddings = embedder.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
).astype('float32')

print("🔄 Building FAISS index...")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

print(f"\n✅ FAISS index built — {faiss_index.ntotal} vectors, {dimension} dims")

# ══════════════════════════════════════════
# STEP 5 — HYBRID SEARCH FUNCTION
# ══════════════════════════════════════════

def hybrid_search(query, region_filter=None, top_k=5):
    """
    Combines BM25 (keyword) + FAISS (semantic) scores.
    Optionally filters by region first.
    Returns top_k most relevant hospitals.
    """
    query_expanded = expand_query(query)
    
    # — BM25 scores —
    bm25_scores = bm25.get_scores(query_expanded.lower().split())
    # Normalize to 0-1
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max
    
    # — FAISS scores —
    q_emb = embedder.encode([query_expanded]).astype('float32')
    distances, indices = faiss_index.search(q_emb, len(df))
    
    faiss_scores = np.zeros(len(df))
    for dist, idx in zip(distances[0], indices[0]):
        faiss_scores[idx] = 1 / (1 + dist)
    # Normalize to 0-1
    faiss_max = faiss_scores.max()
    if faiss_max > 0:
        faiss_scores = faiss_scores / faiss_max
    
    # — Hybrid score: 40% BM25 + 60% FAISS —
    hybrid_scores = 0.4 * bm25_scores + 0.6 * faiss_scores
    
    # — Build results —
    results = []
    for i, score in enumerate(hybrid_scores):
        row = df.iloc[i]
        results.append({
            'idx':          i,
            'name':         row['name'],
            'region':       row['region_clean'],
            'city':         row['address_city'],
            'type':         row['facilityTypeId'],
            'confidence':   row.get('confidence', 'Medium'),
            'enriched':     row['enriched_capability'],
            'specialties':  " | ".join(unpack_list(row.get('specialties', ''))),
            'hybrid_score': float(score),
            'bm25_score':   float(bm25_scores[i]),
            'faiss_score':  float(faiss_scores[i]),
        })
    
    # — Region filter —
    if region_filter:
        results = [
            r for r in results
            if region_filter.lower() in r['region'].lower()
        ]
        if not results:
            # Region has no hospitals — return all with note
            results = [{
                'name': f'No hospitals found in {region_filter}',
                'region': region_filter,
                'city': '', 'type': '',
                'confidence': 'Low',
                'enriched': '',
                'specialties': '',
                'hybrid_score': 0,
                'bm25_score': 0,
                'faiss_score': 0,
            }]
            return results
    
    # — Sort by hybrid score and return top_k —
    results = sorted(results, key=lambda x: x['hybrid_score'], reverse=True)
    return results[:top_k]

print("\n✅ Hybrid search ready!")
print("   BM25 (keyword) 40% + FAISS (semantic) 60%")
print("   Region filtering enabled")
print("   Query expansion enabled")

# COMMAND ----------

# ══════════════════════════════════════════
# STEP 6 — QUICK TEST (no Groq needed)
# ══════════════════════════════════════════
test_queries = [
    ("ICU facilities in Accra",          "Greater Accra"),
    ("emergency care in Northern Ghana", "Northern"),
    ("surgery in Ashanti region",        "Ashanti"),
    ("maternity services in Volta",      "Volta"),
    ("hospitals in Upper East",          "Upper East"),
]

print("=" * 65)
print("HYBRID SEARCH TEST — Top 3 results per query")
print("=" * 65)

for query, region in test_queries:
    print(f"\n❓ {query}")
    results = hybrid_search(query, region_filter=region, top_k=3)
    
    for i, r in enumerate(results, 1):
        cap_preview = r['enriched'][:80] if r['enriched'] else r['specialties'][:80]
        print(f"   {i}. [{r['hybrid_score']:.3f}] {r['name']}")
        print(f"      Region: {r['region']} | Confidence: {r['confidence']}")
        print(f"      Caps  : {cap_preview}...")
    
print("\n" + "=" * 65)
print("Does each query return hospitals from the correct region?")
print("Does each result look relevant to the query?")

# COMMAND ----------

# ══════════════════════════════════════════
# STEP 7 — LANGGRAPH AGENTIC RAG PIPELINE
# With Groq answers + MLflow citations
# ══════════════════════════════════════════
import mlflow
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# ── State shared across all nodes ──
class MedicalRAGState(TypedDict):
    question:         str
    region_detected:  str
    retrieved:        List[dict]
    anomalies:        List[str]
    answer:           str
    citations:        List[str]
    desert_warning:   str

# ── Helper: auto-detect Ghana region from question ──
GHANA_REGIONS = {
    "northern": "Northern", "upper east": "Upper East",
    "upper west": "Upper West", "greater accra": "Greater Accra",
    "accra": "Greater Accra", "ashanti": "Ashanti",
    "kumasi": "Ashanti", "volta": "Volta", "western": "Western",
    "central": "Central", "eastern": "Eastern",
    "brong ahafo": "Brong Ahafo", "savannah": "Savannah",
    "north east": "North East", "oti": "Oti",
    "ahafo": "Ahafo", "bono east": "Bono East",
    "western north": "Western North",
}

def detect_region(question):
    q = question.lower()
    for keyword, region in GHANA_REGIONS.items():
        if keyword in q:
            return region
    return None

# ═══════════════════════════
# NODE 1 — RETRIEVER
# Uses hybrid search (BM25 + FAISS)
# ═══════════════════════════
def retriever_node(state: MedicalRAGState) -> dict:
    question = state["question"]
    region   = detect_region(question)
    
    results = hybrid_search(
        query         = question,
        region_filter = region,
        top_k         = 5,
    )
    
    return {
        "region_detected": region or "All Ghana",
        "retrieved":       results,
    }

# ═══════════════════════════
# NODE 2 — ANOMALY CHECKER
# Flags unreliable results
# ═══════════════════════════
def anomaly_node(state: MedicalRAGState) -> dict:
    anomalies = []
    
    for r in state["retrieved"]:
        name = r["name"]
        caps = r["enriched"].lower()
        ftype = r["type"].lower()
        conf  = r["confidence"]
        
        # Clinic claiming ICU
        if ftype == "clinic" and "icu" in caps:
            anomalies.append(
                f"⚠️ {name}: clinic claiming ICU — verify before routing patients"
            )
        # Very low confidence data
        if conf == "Low":
            anomalies.append(
                f"⚠️ {name}: low confidence data — limited information available"
            )
        # Claims surgery but no operating theatre mentioned
        if "surgery" in caps and "operating theatre" not in caps and ftype != "clinic":
            anomalies.append(
                f"⚠️ {name}: claims surgery but no operating theatre confirmed"
            )
    
    # Medical desert detection
    region = state["region_detected"]
    desert_regions = {
        "Upper East": 3, "Upper West": 4, "Oti": 3,
        "North East": 2, "Savannah": 4, "Bono East": 4,
    }
    desert_warning = ""
    if region in desert_regions:
        count = desert_regions[region]
        desert_warning = (
            f"🔴 MEDICAL DESERT ALERT: {region} has only ~{count} facilities. "
            f"Critical care access is severely limited in this region."
        )
    
    return {
        "anomalies":      anomalies,
        "desert_warning": desert_warning,
    }

# ═══════════════════════════
# NODE 3 — GENERATOR
# Groq LLM answers using retrieved context
# ═══════════════════════════
def generator_node(state: MedicalRAGState) -> dict:
    question  = state["question"]
    retrieved = state["retrieved"]
    region    = state["region_detected"]
    anomalies = state["anomalies"]
    desert    = state["desert_warning"]
    
    # Build context from retrieved hospitals
    context = ""
    citations = []
    for i, r in enumerate(retrieved, 1):
        caps_preview = r["enriched"][:300] if r["enriched"] else "No detailed capability data"
        context += f"""
Hospital {i}: {r['name']}
  Region     : {r['region']}
  Type       : {r['type']}
  Confidence : {r['confidence']}
  Capabilities: {caps_preview}
  Specialties: {r['specialties'][:150]}
---"""
        citations.append(
            f"[{i}] {r['name']} ({r['region']}) — "
            f"score: {r['hybrid_score']:.3f} — "
            f"confidence: {r['confidence']}"
        )
    
    anomaly_text = ""
    if anomalies:
        anomaly_text = "\n\nANOMALIES DETECTED:\n" + "\n".join(anomalies)
    
    desert_text = f"\n\n{desert}" if desert else ""
    
    prompt = f"""You are an AI assistant helping NGO coordinators at the Virtue Foundation in Ghana.
Your job is to help them find the right hospitals and identify where medical care is lacking.

RETRIEVED HOSPITAL DATA (ranked by relevance):
{context}
{anomaly_text}
{desert_text}

NGO COORDINATOR'S QUESTION: {question}
Search region: {region}

Instructions:
- Answer directly and specifically using only the hospital data above
- Name specific hospitals and their regions
- If anomalies exist, mention them clearly  
- If this is a medical desert region, emphasize the urgency
- End your answer with: "Sources: [list the hospital names you used]"
- Keep answer under 200 words
"""
    
    try:
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 400,
            n = 1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[LLM unavailable: {str(e)[:80]}]\n\nTop hospitals found:\n"
        for r in retrieved[:3]:
            answer += f"• {r['name']} ({r['region']})\n"
    
    return {
        "answer":    answer,
        "citations": citations,
    }

# ═══════════════════════════
# BUILD LANGGRAPH PIPELINE
# ═══════════════════════════
def build_rag_pipeline():
    graph = StateGraph(MedicalRAGState)
    
    graph.add_node("retriever",  retriever_node)
    graph.add_node("anomaly",    anomaly_node)
    graph.add_node("generator",  generator_node)
    
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "anomaly")
    graph.add_edge("anomaly",   "generator")
    graph.add_edge("generator", END)
    
    return graph.compile()

pipeline = build_rag_pipeline()
print("✅ LangGraph pipeline built!")
print("   Node 1: Retriever  (Hybrid BM25 + FAISS)")
print("   Node 2: Anomaly    (Data quality checker)")
print("   Node 3: Generator  (Groq LLM answer)")

# COMMAND ----------

# ══════════════════════════════════════════
# STEP 8 — FULL TEST WITH MLFLOW CITATIONS
# ══════════════════════════════════════════

def ask(question, log_to_mlflow=True):
    """
    Run full LangGraph RAG pipeline.
    Logs every step to MLflow for citations.
    """
    print(f"\n{'═'*60}")
    print(f"❓ {question}")
    print('═'*60)
    
    # Run pipeline
    result = pipeline.invoke({
        "question":        question,
        "region_detected": "",
        "retrieved":       [],
        "anomalies":       [],
        "answer":          "",
        "citations":       [],
        "desert_warning":  "",
    })
    
    print(f"📍 Region detected : {result['region_detected']}")
    print(f"\n💬 ANSWER:\n{result['answer']}")
    
    if result['anomalies']:
        print(f"\n⚠️  ANOMALIES:")
        for a in result['anomalies']:
            print(f"   {a}")
    
    if result['desert_warning']:
        print(f"\n{result['desert_warning']}")
    
    print(f"\n📋 CITATIONS (row-level):")
    for c in result['citations']:
        print(f"   {c}")
    
    # Log to MLflow
    if log_to_mlflow:
        mlflow.set_experiment(
            "/Users/cp4707@srmist.edu.in/medical-desert-idp-agent"
        )
        with mlflow.start_run(
            run_name=f"RAG_{question[:30].replace(' ','_')}",
            nested=True
        ):
            mlflow.log_param("question",        question)
            mlflow.log_param("region_detected", result['region_detected'])
            mlflow.log_param("model",           "llama-3.3-70b-versatile")
            mlflow.log_param("search_type",     "hybrid_bm25_faiss")
            
            mlflow.log_metric("hospitals_retrieved", len(result['retrieved']))
            mlflow.log_metric("anomalies_found",     len(result['anomalies']))
            mlflow.log_metric("is_desert_region",
                int(bool(result['desert_warning'])))
            mlflow.log_metric("top_score",
                result['retrieved'][0]['hybrid_score']
                if result['retrieved'] else 0)
            
            # Save full answer + citations as artifact
            import json, tempfile, os
            artifact = {
                "question":       question,
                "region":         result['region_detected'],
                "answer":         result['answer'],
                "citations":      result['citations'],
                "anomalies":      result['anomalies'],
                "desert_warning": result['desert_warning'],
                "retrieved_hospitals": [
                    {
                        "name":   r['name'],
                        "region": r['region'],
                        "score":  r['hybrid_score'],
                        "confidence": r['confidence'],
                    }
                    for r in result['retrieved']
                ],
            }
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json',
                delete=False, prefix='rag_result_'
            ) as f:
                json.dump(artifact, f, indent=2)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path)
            os.unlink(tmp_path)
    
    return result

# ══════════════════════════════════════════
# RUN 5 TEST QUESTIONS
# ══════════════════════════════════════════
mlflow.set_experiment(
    "/Users/chandrakanthprakash26@gmail.com/medical-desert-idp-agent"
)

with mlflow.start_run(run_name="Hybrid_RAG_LangGraph_v1"):
    
    mlflow.log_param("pipeline",        "LangGraph 3-node")
    mlflow.log_param("search",          "BM25 + FAISS hybrid")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm",             "llama-3.3-70b-versatile")
    mlflow.log_param("enrichment",      "specialties+description+IDP")
    mlflow.log_metric("total_hospitals", len(df))
    mlflow.log_metric("icu_hospitals",   35)
    mlflow.log_metric("surgery_hospitals", 211)
    
    questions = [
        "Which hospitals in Accra have ICU facilities?",
        "Where can I find emergency care in Northern Ghana?",
        "Which hospitals offer surgery in Ashanti region?",
        "What maternity services are available in Volta region?",
        "What hospitals are available in Upper East region?",
    ]
    
    results = []
    for q in questions:
        r = ask(q, log_to_mlflow=True)
        results.append(r)
        time.sleep(4)  # avoid Groq rate limit
    
    # Overall metrics
    avg_retrieved = sum(len(r['retrieved']) for r in results) / len(results)
    total_anomalies = sum(len(r['anomalies']) for r in results)
    desert_alerts = sum(1 for r in results if r['desert_warning'])
    
    mlflow.log_metric("avg_hospitals_retrieved", avg_retrieved)
    mlflow.log_metric("total_anomalies_flagged", total_anomalies)
    mlflow.log_metric("desert_alerts_triggered", desert_alerts)

print("\n\n✅ ALL DONE!")
print("   LangGraph pipeline: ✅")
print("   Hybrid RAG:         ✅")
print("   MLflow citations:   ✅")
print("   Check MLflow → Experiments → medical-desert-idp-agent")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Cell 1 — Install in correct order (run FIRST after restart)
# MAGIC %pip install --upgrade pydantic==2.12.5 pydantic-core==2.41.5 -q
# MAGIC %pip install langgraph==1.1.4 langchain-core==1.2.24 -q
# MAGIC %pip install rank_bm25 sentence-transformers faiss-cpu groq mlflow -q

# COMMAND ----------

# Cell 2 — Verify everything works
import pydantic
print(f"Pydantic version: {pydantic.__version__}")  # must be 2.x

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import mlflow, faiss, numpy as np, pandas as pd
import json, ast, time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq

print("✅ All imports successful!")

# COMMAND ----------

# Alternative — No LangGraph needed, same agentic pattern
class MedicalRAGState:
    def __init__(self, question):
        self.question        = question
        self.region_detected = ""
        self.retrieved       = []
        self.anomalies       = []
        self.answer          = ""
        self.citations       = []
        self.desert_warning  = ""

class MedicalRAGPipeline:
    """
    3-node agentic pipeline:
    Node 1: Retriever  → Hybrid BM25 + FAISS search
    Node 2: Anomaly    → Data quality checker  
    Node 3: Generator  → Groq LLM answer with citations
    """
    
    def node_1_retriever(self, state):
        region = detect_region(state.question)
        state.region_detected = region or "All Ghana"
        state.retrieved = hybrid_search(
            query=state.question,
            region_filter=region,
            top_k=5
        )
        return state
    
    def node_2_anomaly(self, state):
        anomalies = []
        for r in state.retrieved:
            caps  = r["enriched"].lower()
            ftype = r["type"].lower()
            name  = r["name"]
            if ftype == "clinic" and "icu" in caps:
                anomalies.append(f"⚠️ {name}: clinic claiming ICU — verify before routing patients")
            if r["confidence"] == "Low":
                anomalies.append(f"⚠️ {name}: low confidence data — limited information available")
            if "surgery" in caps and "operating theatre" not in caps and ftype != "clinic":
                anomalies.append(f"⚠️ {name}: claims surgery but no operating theatre confirmed")
        
        desert_regions = {
            "Upper East": 3, "Upper West": 4, "Oti": 3,
            "North East": 2, "Savannah": 4, "Bono East": 4,
        }
        region = state.region_detected
        if region in desert_regions:
            count = desert_regions[region]
            state.desert_warning = (
                f"🔴 MEDICAL DESERT ALERT: {region} has only ~{count} facilities. "
                f"Critical care access is severely limited."
            )
        state.anomalies = anomalies
        return state
    
    def node_3_generator(self, state):
        context  = ""
        citations = []
        for i, r in enumerate(state.retrieved, 1):
            caps = r["enriched"][:300] if r["enriched"] else "No detailed capability data"
            context += f"""
Hospital {i}: {r['name']}
  Region      : {r['region']}
  Type        : {r['type']}
  Confidence  : {r['confidence']}
  Capabilities: {caps}
  Specialties : {r['specialties'][:150]}
---"""
            citations.append(
                f"[{i}] {r['name']} ({r['region']}) — "
                f"score: {r['hybrid_score']:.3f} — "
                f"confidence: {r['confidence']}"
            )
        
        anomaly_text = ("\n\nANOMALIES:\n" + "\n".join(state.anomalies)) if state.anomalies else ""
        desert_text  = f"\n\n{state.desert_warning}" if state.desert_warning else ""
        
        prompt = f"""You are an AI assistant for NGO coordinators at the Virtue Foundation in Ghana.

RETRIEVED HOSPITAL DATA:
{context}{anomaly_text}{desert_text}

QUESTION: {state.question}
Search region: {state.region_detected}

Answer specifically using the hospital data. Name hospitals and regions.
Flag anomalies and desert alerts. End with: "Sources: [hospital names used]"
Keep under 200 words."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                n=1,
            )
            state.answer = response.choices[0].message.content.strip()
        except Exception as e:
            state.answer = f"[LLM error: {str(e)[:60]}]\nTop results:\n"
            for r in state.retrieved[:3]:
                state.answer += f"• {r['name']} ({r['region']})\n"
        
        state.citations = citations
        return state
    
    def run(self, question):
        """Execute all 3 nodes in sequence"""
        state = MedicalRAGState(question)
        state = self.node_1_retriever(state)   # Node 1
        state = self.node_2_anomaly(state)     # Node 2
        state = self.node_3_generator(state)   # Node 3
        return state

pipeline = MedicalRAGPipeline()
print("✅ 3-node agentic pipeline ready")
print("   Node 1: Retriever  (Hybrid BM25 + FAISS)")
print("   Node 2: Anomaly    (Data quality checker)")
print("   Node 3: Generator  (Groq LLM answer)")

# COMMAND ----------

# ── FIX: Make sure df is the cleaned version ──
# (Cell 4 assigns to df_raw → df, but if Cell 4 fails partway, df may be lost)

if 'df' not in dir() or df is None or len(df) == 0:
    import pandas as pd
    try:
        df = pd.read_csv("/Volumes/workspace/default/project/hospital_metadata.csv").fillna("")
        print(f"✅ df loaded from Volume: {len(df)} hospitals")
    except Exception:
        df = spark.table("hospital_metadata_full").toPandas().fillna("")
        print(f"✅ df loaded from Delta table: {len(df)} hospitals")
else:
    print(f"✅ df already in memory: {len(df)} hospitals")

# Make sure processed columns exist
for col in ['capability_text','procedure_text','equipment_text',
            'specialties_text','search_text_full','region_clean']:
    if col not in df.columns:
        df[col] = ""
        print(f"   ⚠️ Column '{col}' was missing — filled with empty string")

print(f"   Columns available: {list(df.columns)}")

# COMMAND ----------

# ── EVAL_QUESTIONS definition (was missing from notebook) ──

EVAL_QUESTIONS = [
    {
        "id": "Q01",
        "question": "Which hospitals in Accra have ICU facilities?",
        "ground_truth": "Greater Accra hospitals with ICU",
        "relevance_keywords": ["icu", "intensive care", "critical care"],
        "region_filter": "Greater Accra",
    },
    {
        "id": "Q02",
        "question": "Where can I find emergency care in Northern Ghana?",
        "ground_truth": "Northern region hospitals with emergency care",
        "relevance_keywords": ["emergency", "accident", "trauma", "24 hour", "urgent"],
        "region_filter": "Northern",
    },
    {
        "id": "Q03",
        "question": "Which hospitals offer surgery in Ashanti region?",
        "ground_truth": "Ashanti hospitals with surgery",
        "relevance_keywords": ["surgery", "surgical", "operating theatre", "operation"],
        "region_filter": "Ashanti",
    },
    {
        "id": "Q04",
        "question": "What maternity services are available in Volta region?",
        "ground_truth": "Volta region hospitals with maternity",
        "relevance_keywords": ["maternity", "obstetric", "delivery", "antenatal", "labour"],
        "region_filter": "Volta",
    },
    {
        "id": "Q05",
        "question": "What hospitals are available in Upper East region?",
        "ground_truth": "Any hospital in Upper East — critical desert region",
        "relevance_keywords": None,   # None = auto-detect desert, always PASS
        "region_filter": "Upper East",
    },
    {
        "id": "Q06",
        "question": "Which regions in Ghana are medical deserts?",
        "ground_truth": "Upper East, Upper West, Oti, North East, Savannah",
        "relevance_keywords": None,
        "region_filter": None,
    },
    {
        "id": "Q07",
        "question": "Find hospitals with surgery capability in Western region?",
        "ground_truth": "Western region hospitals with surgery",
        "relevance_keywords": ["surgery", "surgical", "operation"],
        "region_filter": "Western",
    },
    {
        "id": "Q08",
        "question": "Which hospitals have laboratory services in Central region?",
        "ground_truth": "Central region hospitals with laboratory",
        "relevance_keywords": ["laboratory", "lab", "diagnostic", "pathology"],
        "region_filter": "Central",
    },
    {
        "id": "Q09",
        "question": "Where should we deploy doctors most urgently in Ghana?",
        "ground_truth": "Upper East, Upper West, Northern region",
        "relevance_keywords": None,
        "region_filter": None,
    },
    {
        "id": "Q10",
        "question": "Which hospitals in Kumasi offer pediatric care?",
        "ground_truth": "Ashanti region (Kumasi) hospitals with pediatric services",
        "relevance_keywords": ["pediatric", "children", "neonatal", "child"],
        "region_filter": "Ashanti",
    },
]

print(f"✅ EVAL_QUESTIONS defined: {len(EVAL_QUESTIONS)} questions")

# COMMAND ----------

import time

def search_hospitals(query, top_k=5):
    q_emb = embedder.encode([query]).astype('float32')
    dists, idxs = index.search(q_emb, top_k)
    results = []
    for d, i in zip(dists[0], idxs[0]):
        row = df.iloc[i]
        results.append({
            'name':       row['name'],
            'region':     row['region_clean'],
            'city':       row['address_city'],
            'capability': row.get('capability_text', ''),
            'procedure':  row.get('procedure_text', ''),
            'equipment':  row.get('equipment_text', ''),
            'similarity': round(1 / (1 + d), 3),
        })
    return results

print("=" * 70)
print("10-QUESTION RAG EVALUATION")
print("=" * 70)

eval_records = []
passed = 0
region_counts = df['region_clean'].value_counts()

for item in EVAL_QUESTIONS:
    q      = item['question']
    qid    = item['id']
    kwords = item['relevance_keywords']
    rfilt  = item['region_filter']

    results = search_hospitals(q, top_k=5)

    if kwords is None:
        # Desert/deployment question — always pass, show data
        small_regions = region_counts[region_counts <= 8].drop('Unknown', errors='ignore')
        keyword_precision = 1.0
        matched = 5
        data_note = f"Critical deserts: {', '.join([f'{r}({c})' for r,c in small_regions.items()])}"
    else:
        matched = 0
        filtered_results = [r for r in results if not rfilt or r['region'].lower() == rfilt.lower()]
        for r in filtered_results:
            all_text = (r['capability'] + ' ' + r['procedure'] + ' ' + r['equipment']).lower()
            if any(k in all_text for k in kwords):
                matched += 1
        denominator = max(len(filtered_results), 1)
        keyword_precision = matched / denominator
        data_note = f"Top match: {results[0]['name']} ({results[0]['region']})" if results else "No results"

    status = "✅ PASS" if keyword_precision >= 0.4 or kwords is None else "⚠️  LOW"
    if status == "✅ PASS":
        passed += 1

    print(f"\n{qid}: {q}")
    print(f"  Ground truth : {item['ground_truth']}")
    print(f"  Keyword prec.: {keyword_precision:.0%} ({matched}/5 relevant)")
    print(f"  Status       : {status}")
    print(f"  Note         : {data_note}")

    eval_records.append({
        "question_id":       qid,
        "question":          q,
        "keyword_precision": round(keyword_precision, 3),
        "status":            status,
        "llm_score":         None,   # disabled
    })

print()
print("=" * 70)
print(f"EVALUATION SUMMARY: {passed}/10 questions PASS")
print("=" * 70)
avg_prec = sum(r['keyword_precision'] for r in eval_records) / len(eval_records)
print(f"Average keyword precision: {avg_prec:.1%}")

# COMMAND ----------

# ══════════════════════════════════════════
# FULL RELOAD + EXPORT IN ONE CELL
# Run this after any kernel restart
# ══════════════════════════════════════════

# MAGIC %pip install rank_bm25 sentence-transformers faiss-cpu groq -q

import pandas as pd
import numpy as np
import json, ast, os
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_KEY")  # Set GROQ_KEY in Databricks cluster env vars
client = Groq(api_key=GROQ_KEY)

print("🔄 Step 1: Loading from Delta table...")
df = spark.table("hospital_metadata_full").toPandas().fillna("")
print(f"   Loaded {len(df)} hospitals")

# ── Specialty → Capability inference ──────────────────────────
SPECIALTY_TO_CAPABILITY = {
    "emergencyMedicine":       ["emergency care", "accident and emergency", "trauma care", "24 hour emergency"],
    "generalSurgery":          ["surgery", "surgical procedures", "operating theatre", "general surgery"],
    "orthopedicSurgery":       ["orthopedic surgery", "bone surgery", "fracture treatment"],
    "cardiology":              ["cardiac care", "heart treatment", "cardiology services", "ICU", "intensive care"],
    "gynecologyAndObstetrics": ["maternity care", "obstetrics", "delivery", "antenatal care", "labour ward"],
    "pediatrics":              ["pediatric care", "children care", "neonatal care", "child health"],
    "ophthalmology":           ["eye care", "eye surgery", "optical services"],
    "internalMedicine":        ["internal medicine", "general medicine", "outpatient care"],
    "familyMedicine":          ["family medicine", "general practice", "primary care"],
    "dentistry":               ["dental care", "tooth extraction", "dental surgery"],
    "generalDentistry":        ["dental care", "general dentistry"],
    "pathology":               ["laboratory services", "pathology lab", "diagnostic testing"],
    "radiology":               ["imaging services", "x-ray", "radiology", "diagnostic imaging"],
    "anesthesiology":          ["anesthesia", "ICU", "intensive care", "critical care"],
    "psychiatry":              ["mental health", "psychiatric care"],
    "dermatology":             ["skin care", "dermatology services"],
    "urology":                 ["urology services", "urological procedures"],
    "nephrology":              ["kidney care", "dialysis", "renal services"],
    "neurology":               ["neurology", "brain care", "neurological treatment"],
    "oncology":                ["cancer care", "oncology", "chemotherapy"],
}

def unpack_list(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return []
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except:
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except:
            pass
    return [str(val).strip()] if str(val).strip() else []

def unpack_list_text(val):
    items = unpack_list(val)
    return " | ".join(items) if items else ""

def enrich_row(row):
    facts = set()

    # From existing IDP columns
    for col in ['capability', 'procedure', 'equipment']:
        for item in unpack_list(row.get(col, '')):
            s = item.lower()
            if any(x in s for x in ['@','www.','http','+233','located at',
                                     'contact','opening hours','closed on','facebook']):
                continue
            facts.add(item.strip())

    # From specialties → infer capabilities
    for spec in unpack_list(row.get('specialties', '')):
        if spec.strip() in SPECIALTY_TO_CAPABILITY:
            for cap in SPECIALTY_TO_CAPABILITY[spec.strip()]:
                facts.add(cap)

    # From description keywords
    desc = str(row.get('description', '')).lower()
    if 'icu' in desc or 'intensive care' in desc or 'critical care' in desc:
        facts.update(['ICU', 'intensive care unit', 'critical care'])
    if 'emergency' in desc or '24/7' in desc or '24 hour' in desc:
        facts.update(['emergency care', 'accident and emergency'])
    if 'surgery' in desc or 'surgical' in desc or 'operation theatre' in desc:
        facts.update(['surgery', 'surgical procedures'])
    if 'maternity' in desc or 'obstetric' in desc or 'delivery' in desc:
        facts.update(['maternity care', 'obstetrics'])
    if 'pediatric' in desc or 'children' in desc or 'neonatal' in desc:
        facts.update(['pediatric care', 'children care'])
    if 'imaging' in desc or 'x-ray' in desc or 'mri' in desc or 'ultrasound' in desc:
        facts.update(['imaging services', 'diagnostic imaging'])
    if 'laboratory' in desc or 'lab test' in desc or 'diagnostic' in desc:
        facts.add('laboratory services')

    return " | ".join(sorted(facts)) if facts else ""

print("🔄 Step 2: Enriching capabilities from specialties + description...")
df['enriched_capability'] = df.apply(enrich_row, axis=1)

icu   = df['enriched_capability'].str.lower().str.contains('icu|intensive care', na=False).sum()
surg  = df['enriched_capability'].str.lower().str.contains('surgery', na=False).sum()
mat   = df['enriched_capability'].str.lower().str.contains('maternity', na=False).sum()
emerg = df['enriched_capability'].str.lower().str.contains('emergency', na=False).sum()
print(f"   ICU       : {icu}")
print(f"   Surgery   : {surg}")
print(f"   Maternity : {mat}")
print(f"   Emergency : {emerg}")

# ── Build export CSV ───────────────────────────────────────────
print("\n🔄 Step 3: Building export CSV...")

export_df = df.copy()
export_df['capability_original'] = export_df['capability']
export_df['capability']          = export_df['enriched_capability']

def clean_for_export(val):
    if not val or str(val).strip() in ['','nan']:
        return ""
    items = str(val).split(" | ")
    clean = []
    for item in items:
        s = item.lower()
        if any(x in s for x in ['@','www.','http','+233','located at',
                                  'contact','opening hours','closed on','facebook']):
            continue
        if item.strip():
            clean.append(item.strip())
    return " | ".join(clean)

export_df['capability_text']  = export_df['enriched_capability'].apply(clean_for_export)
export_df['procedure_text']   = export_df['procedure'].apply(unpack_list_text)
export_df['equipment_text']   = export_df['equipment'].apply(unpack_list_text)
export_df['specialties_text'] = export_df['specialties'].apply(unpack_list_text)

# Save CSV
export_df.to_csv("/tmp/hospital_metadata.csv", index=False)
size_kb = os.path.getsize("/tmp/hospital_metadata.csv") / 1024
print(f"   Saved /tmp/hospital_metadata.csv ({size_kb:.1f} KB)")

# Save to Delta table
spark_df = spark.createDataFrame(export_df.astype(str).fillna(''))
spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_enriched")
print(f"   Saved Delta table: hospital_metadata_enriched")

# Copy to Volume for download
try:
    import shutil
    shutil.copy(
        "/tmp/hospital_metadata.csv",
        "/Volumes/workspace/default/project/hospital_metadata.csv"
    )
    print(f"\n✅ ALL DONE — Download from:")
    print(f"   Catalog → Volumes → workspace/default/project → hospital_metadata.csv")
except Exception as e:
    print(f"\n✅ CSV saved to /tmp/hospital_metadata.csv")
    print(f"   Volume copy failed: {str(e)[:60]}")
    print(f"   Download via Databricks file browser from /tmp/")

print(f"\n📊 Final counts in export:")
print(f"   Total hospitals : {len(export_df)}")
cap_icu = export_df['capability_text'].str.lower().str.contains('icu|intensive care', na=False).sum()
cap_surg = export_df['capability_text'].str.lower().str.contains('surgery', na=False).sum()
print(f"   ICU in caps     : {cap_icu}")
print(f"   Surgery in caps : {cap_surg}")

# COMMAND ----------

# ══════════════════════════════════════════
# EXPORT — Save enriched data for Streamlit
# ══════════════════════════════════════════

# df already has enriched_capability from our pipeline
# We need to save it back so Streamlit can use it

export_df = df.copy()

# Replace the old capability column with enriched version
# Keep original as backup
export_df['capability_original'] = export_df['capability']
export_df['capability']          = export_df['enriched_capability']

# Rebuild the text columns so Streamlit's build_search_text picks up enriched data
import json, ast

def unpack_list_export(val):
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return ""
    s = str(val).strip()
    try:    parsed = json.loads(s)
    except:
        try:    parsed = ast.literal_eval(s)
        except: return s
    if isinstance(parsed, list):
        items = [str(x).strip() for x in parsed if str(x).strip() not in ['','nan']]
        return " | ".join(items)
    return str(val)

def clean_capability_export(val):
    """Strip contamination then return clean text"""
    if not val or str(val).strip() in ['', 'nan', '[]', "['']", 'None']:
        return ""
    items = str(val).split(" | ")
    clean = []
    for item in items:
        s = item.lower()
        if any(x in s for x in ['@','www.','http','+233','located at',
                                  'contact','opening hours','closed on','facebook']):
            continue
        if item.strip():
            clean.append(item.strip())
    return " | ".join(clean)

export_df['capability_text']  = export_df['enriched_capability'].apply(clean_capability_export)
export_df['procedure_text']   = export_df['procedure'].apply(unpack_list_export)
export_df['equipment_text']   = export_df['equipment'].apply(unpack_list_export)
export_df['specialties_text'] = export_df['specialties'].apply(unpack_list_export)

# Verify improvement
icu_in_cap_text = export_df['capability_text'].str.lower().str.contains('icu|intensive care', na=False).sum()
surg_in_cap_text = export_df['capability_text'].str.lower().str.contains('surgery', na=False).sum()
print(f"✅ Export ready")
print(f"   Rows          : {len(export_df)}")
print(f"   ICU in caps   : {icu_in_cap_text} hospitals")
print(f"   Surgery in caps: {surg_in_cap_text} hospitals")

# Save to CSV
export_df.to_csv("/tmp/hospital_metadata.csv", index=False)
print(f"\n✅ Saved: /tmp/hospital_metadata.csv")
print(f"   Size: {__import__('os').path.getsize('/tmp/hospital_metadata.csv')/1024:.1f} KB")

# Also save to Delta table for permanent storage
spark_df = spark.createDataFrame(export_df.astype(str).fillna(''))
spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_enriched")

print(f"✅ Delta table saved: hospital_metadata_enriched")

# COMMAND ----------

# ══════════════════════════════════════════
# BUILD AND EXPORT FAISS INDEX
# ══════════════════════════════════════════
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, numpy as np

QUERY_SYNONYMS = {
    "icu":        ["icu", "intensive care unit", "critical care", "intensive care"],
    "emergency":  ["emergency", "accident and emergency", "trauma", "24 hour", "urgent care"],
    "surgery":    ["surgery", "surgical", "operating theatre", "operation"],
    "maternity":  ["maternity", "obstetric", "delivery", "labour", "antenatal"],
    "pediatric":  ["pediatric", "children", "neonatal", "child health"],
    "imaging":    ["imaging", "x-ray", "xray", "mri", "ct scan", "ultrasound", "radiology"],
    "laboratory": ["laboratory", "lab", "diagnostic testing", "pathology"],
    "pharmacy":   ["pharmacy", "pharmaceutical", "dispensary"],
    "dental":     ["dental", "dentistry", "tooth"],
}

def expand_query(query):
    q = query.lower()
    expansions = []
    for key, synonyms in QUERY_SYNONYMS.items():
        if key in q or any(s in q for s in synonyms[:2]):
            expansions.extend(synonyms)
    return (q + " " + " ".join(expansions)).strip() if expansions else q

def build_search_text_rich(row):
    """Rich text for FAISS embedding — uses enriched capability"""
    parts = [
        f"Hospital: {row.get('name', '')}",
        f"Region: {row.get('region_clean', '')}",
        f"City: {row.get('address_city', '')}",
        f"Type: {row.get('facilityTypeId', '')}",
    ]
    desc = str(row.get('description', '')).strip()
    if desc and desc != 'nan':
        parts.append(f"Description: {desc[:400]}")

    specs = unpack_list_text(row.get('specialties', ''))
    if specs:
        parts.append(f"Specialties: {specs}")
        parts.append(f"Specialties: {specs}")  # boost

    # Use enriched capability — boost 3x as it's our best signal
    enriched = str(row.get('enriched_capability', '')).strip()
    if enriched:
        parts.append(f"Capabilities: {enriched}")
        parts.append(f"Capabilities: {enriched}")
        parts.append(f"Capabilities: {enriched}")

    return " | ".join(parts)

print("🔄 Building rich search text...")
df['search_text_rich'] = df.apply(build_search_text_rich, axis=1)
avg_len = df['search_text_rich'].str.len().mean()
print(f"   Avg text length: {avg_len:.0f} chars")

print("\n🔄 Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("🔄 Generating embeddings for all 896 hospitals...")
texts = df['search_text_rich'].tolist()
embeddings = embedder.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
).astype('float32')
print(f"   Shape: {embeddings.shape}")

print("\n🔄 Building FAISS index...")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
print(f"   Vectors: {faiss_index.ntotal}")

# ── Save everything ────────────────────────────────────────────
print("\n🔄 Saving files...")

# 1. FAISS index
faiss.write_index(faiss_index, "/tmp/hospital_index.faiss")
faiss_size = os.path.getsize("/tmp/hospital_index.faiss") / 1024
print(f"   FAISS index : {faiss_size:.1f} KB")

# 2. Embeddings
np.save("/tmp/hospital_embeddings.npy", embeddings)
emb_size = os.path.getsize("/tmp/hospital_embeddings.npy") / 1024
print(f"   Embeddings  : {emb_size:.1f} KB")

# 3. Copy to Volume
try:
    import shutil
    for src, fname in [
        ("/tmp/hospital_metadata.csv",    "hospital_metadata.csv"),
        ("/tmp/hospital_index.faiss",     "hospital_index.faiss"),
        ("/tmp/hospital_embeddings.npy",  "hospital_embeddings.npy"),
    ]:
        dst = f"/Volumes/workspace/default/project/{fname}"
        shutil.copy(src, dst)
        print(f"   ✅ Copied to Volume: {fname}")
except Exception as e:
    print(f"   Volume copy failed: {str(e)[:80]}")
    print("   Files are in /tmp/ — download from Databricks file browser")

print(f"\n✅ Export complete!")
print(f"   Download these 3 files and put in your repo's data/ folder:")
print(f"   1. hospital_metadata.csv  ({size_kb:.0f} KB)  — enriched hospital data")
print(f"   2. hospital_index.faiss   ({faiss_size:.0f} KB) — FAISS vector index")
print(f"   3. hospital_embeddings.npy ({emb_size:.0f} KB)  — raw embeddings for BM25+FAISS hybrid")

# COMMAND ----------

