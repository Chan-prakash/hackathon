# Databricks notebook source
import pandas as pd

df = pd.read_csv("/Volumes/workspace/default/project/Virtue Foundation Ghana v0.3 - Sheet1.csv")
print(f"✅ File found! Total facilities: {len(df)}")
print(f"📋 Columns: {list(df.columns[:5])}...")

# COMMAND ----------

# Understand what we have - fill rates for key columns
key_cols = ['name', 'address_city', 'address_stateOrRegion', 
            'facilityTypeId', 'description', 'procedure', 
            'equipment', 'capability', 'specialties']

print("=== HOW COMPLETE IS OUR DATA? ===\n")
for col in key_cols:
    filled = df[col].notna().sum()
    empty = len(df) - filled
    pct = round(filled/len(df)*100, 1)
    bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
    print(f"{col:<25} [{bar}] {pct}% filled ({empty} missing)")

# COMMAND ----------

# Normalize messy region names
# "Greater Accra", "ACCRA", "Greater Accra Region" → all become "Greater Accra"

region_mapping = {
    'greater accra region': 'Greater Accra',
    'greater accra':        'Greater Accra',
    'accra':                'Greater Accra',
    'east legon':           'Greater Accra',
    'tema':                 'Greater Accra',
    'ashanti region':       'Ashanti',
    'ashanti':              'Ashanti',
    'kumasi':               'Ashanti',
    'western region':       'Western',
    'western':              'Western',
    'western north region': 'Western North',
    'northern region':      'Northern',
    'northern':             'Northern',
    'tamale':               'Northern',
    'volta region':         'Volta',
    'volta':                'Volta',
    'central region':       'Central',
    'central':              'Central',
    'eastern region':       'Eastern',
    'eastern':              'Eastern',
    'brong ahafo region':   'Brong Ahafo',
    'brong ahafo':          'Brong Ahafo',
    'bono east region':     'Bono East',
    'upper east region':    'Upper East',
    'upper east':           'Upper East',
    'upper west region':    'Upper West',
    'upper west':           'Upper West',
    'savannah region':      'Savannah',
    'north east region':    'North East',
    'oti region':           'Oti',
    'ahafo region':         'Ahafo',
}

def normalize_region(region):
    if pd.isna(region):
        return 'Unknown'
    return region_mapping.get(region.strip().lower(), region.strip())

df['region_clean'] = df['address_stateOrRegion'].apply(normalize_region)

print("=== CLEANED REGION COUNTS ===\n")
region_counts = df['region_clean'].value_counts()
for region, count in region_counts.items():
    bar = "█" * count
    print(f"{region:<20} {bar} ({count})")

# COMMAND ----------

# Flag each hospital row with what's missing or needs AI extraction

def check_quality(row):
    flags = []
    
    # No description at all
    if pd.isna(row['description']) or str(row['description']).strip() == '':
        flags.append('NO_DESCRIPTION')
    
    # Check if procedure/equipment/capability are empty
    proc_empty  = pd.isna(row['procedure'])  or str(row['procedure']).strip()  in ['', '[]', "['']"]
    equip_empty = pd.isna(row['equipment'])  or str(row['equipment']).strip()  in ['', '[]', "['']"]
    cap_empty   = pd.isna(row['capability']) or str(row['capability']).strip() in ['', '[]', "['']"]
    has_desc    = not pd.isna(row['description']) and str(row['description']).strip() != ''
    
    # Has description but AI hasn't extracted facts yet → our main job!
    if has_desc and proc_empty and equip_empty:
        flags.append('NEEDS_IDP_EXTRACTION')
    
    # Missing location info
    if pd.isna(row['address_city']) and pd.isna(row['address_stateOrRegion']):
        flags.append('NO_LOCATION')
    
    # Missing facility type
    if pd.isna(row['facilityTypeId']):
        flags.append('NO_FACILITY_TYPE')
    
    return ', '.join(flags) if flags else 'COMPLETE'

df['quality_flags'] = df.apply(check_quality, axis=1)

# Summary
from collections import Counter
all_flags = []
for f in df['quality_flags']:
    all_flags.extend(f.split(', '))
flag_counts = Counter(all_flags)

print("=== DATA QUALITY SUMMARY ===\n")
for flag, count in flag_counts.most_common():
    print(f"  {flag:<30} → {count} facilities")

# COMMAND ----------

# Which regions look like medical deserts RIGHT NOW?

import numpy as np

region_summary = df.groupby('region_clean').agg(
    total_facilities = ('name', 'count'),
    has_procedure    = ('procedure', lambda x: (x.notna() & (x != '[]')).sum()),
    has_equipment    = ('equipment', lambda x: (x.notna() & (x != '[]')).sum()),
    has_description  = ('description', lambda x: x.notna().sum()),
).reset_index().sort_values('total_facilities', ascending=False)

def risk_level(row):
    if row['total_facilities'] <= 3:
        return '🔴 CRITICAL DESERT'
    elif row['total_facilities'] <= 8:
        return '🟠 HIGH RISK'
    elif row['total_facilities'] <= 20:
        return '🟡 MODERATE'
    else:
        return '🟢 WELL COVERED'

region_summary['risk'] = region_summary.apply(risk_level, axis=1)

print("=== MEDICAL DESERT ANALYSIS (PREVIEW) ===\n")
print(f"{'Region':<22} {'Facilities':>12} {'Has Procedure':>14} {'Has Equipment':>14} {'Risk'}")
print("-" * 80)
for _, row in region_summary.iterrows():
    print(f"{row['region_clean']:<22} {row['total_facilities']:>12} {row['has_procedure']:>14} {row['has_equipment']:>14}  {row['risk']}")

# COMMAND ----------

# Clean column names (remove spaces and special characters)
df.columns = df.columns.str.replace(" ", "_")

spark_df = spark.createDataFrame(df.astype(str).fillna(''))

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medical_facilities_clean")

print("✅ Delta table saved: medical_facilities_clean")
print(f"Total rows: {spark_df.count()}")

# COMMAND ----------

# TEST ALL 3 FREE AI OPTIONS
# We'll use whichever one works!

results = {}

# ============================================
# TEST 1: Databricks Built-in AI
# ============================================
try:
    response = spark.sql("""
        SELECT ai_query(
            'databricks-meta-llama-3-1-70b-instruct',
            'Say exactly this: DATABRICKS AI WORKS'
        ) as response
    """).collect()[0][0]
    results['databricks'] = f"✅ WORKS! Response: {response}"
except Exception as e:
    results['databricks'] = f"❌ Failed: {str(e)[:100]}"

print("1️⃣ Databricks AI:", results['databricks'])
print()

# ============================================
# TEST 2: Google Gemini (Free)
# ============================================
try:
    import subprocess
    subprocess.run(["pip", "install", "google-generativeai", "-q"], 
                  capture_output=True)
    import google.generativeai as genai
    
    # You need to paste your Gemini key here
    GEMINI_KEY = "paste-gemini-key-here"
    
    if GEMINI_KEY != "paste-gemini-key-here":
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Say exactly this: GEMINI WORKS')
        results['gemini'] = f"✅ WORKS! Response: {response.text}"
    else:
        results['gemini'] = "⏭️ SKIPPED - No key pasted yet"
except Exception as e:
    results['gemini'] = f"❌ Failed: {str(e)[:100]}"

print("2️⃣ Google Gemini:", results['gemini'])
print()

# ============================================
# TEST 3: Groq (Free)
# ============================================
try:
    subprocess.run(["pip", "install", "groq", "-q"], 
                  capture_output=True)
    from groq import Groq
    
    GROQ_KEY = ""
    
    if GROQ_KEY != "paste-groq-key-here":
        client = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", 
                      "content": "Say exactly this: GROQ WORKS"}]
        )
        results['groq'] = f"✅ WORKS! Response: {response.choices[0].message.content}"
    else:
        results['groq'] = "⏭️ SKIPPED - No key pasted yet"
except Exception as e:
    results['groq'] = f"❌ Failed: {str(e)[:100]}"

print("3️⃣ Groq:", results['groq'])
print()

# ============================================
# SUMMARY
# ============================================
print("=" * 50)
print("SUMMARY — Which AI can we use?")
print("=" * 50)
for name, result in results.items():
    status = "✅ USE THIS" if "WORKS" in result else "❌ Skip"
    print(f"  {name.upper():<15} → {status}")

# COMMAND ----------

# Try different Databricks model names
models_to_try = [
    "databricks-dbrx-instruct",
    "databricks-mixtral-8x7b-instruct", 
    "databricks-llama-2-70b-chat",
    "databricks-meta-llama-3-70b-instruct",
]

for model in models_to_try:
    try:
        response = spark.sql(f"""
            SELECT ai_query(
                '{model}',
                'Say: WORKS'
            ) as response
        """).collect()[0][0]
        print(f"✅ {model} → WORKS!")
        break
    except Exception as e:
        print(f"❌ {model} → {str(e)[:80]}")

# COMMAND ----------

# MAGIC %pip install groq sentence-transformers faiss-cpu langgraph langchain langchain-groq folium mlflow -q

# COMMAND ----------

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say: GROQ WORKS"}],
    max_tokens=10
)
print(f"✅ {response.choices[0].message.content}")