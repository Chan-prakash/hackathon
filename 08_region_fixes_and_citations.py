# Databricks notebook source
# MAGIC %pip install groq sentence-transformers faiss-cpu -q

# COMMAND ----------

import pandas as pd

print("=== DELTA TABLES ===\n")
tables = [
    "medical_facilities_clean",
    "enriched_facilities",
    "region_gap_analysis",
    "facility_anomalies",
    "hospital_metadata_full",
]
for table in tables:
    try:
        count = spark.table(table).count()
        print(f"✅ {table:<35} → {count} rows")
    except:
        print(f"❌ {table:<35} → MISSING")

print("\n=== DONE ===")

# COMMAND ----------

# Cell 1 — Diagnose the data quality issues
import pandas as pd

df = spark.table("hospital_metadata_full").toPandas().fillna("")

print(f"=== DATA QUALITY REPORT ===\n")
print(f"Total rows          : {len(df)}")
print(f"Unique hospital names: {df['name'].nunique()}")
print(f"Duplicates          : {len(df) - df['name'].nunique()}")

print(f"\n=== REGION ISSUES ===")
print(f"Unknown regions     : {(df['region_clean'] == 'Unknown').sum()}")
print(f"\nAll region counts:")
print(df['region_clean'].value_counts().to_string())

# COMMAND ----------

# Cell 2 — Backup before we change anything
print("🔄 Creating backup...")

spark.table("hospital_metadata_full") \
     .write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("hospital_metadata_full_backup")

count = spark.table("hospital_metadata_full_backup").count()
print(f"✅ Backup created: {count} rows")
print(f"   Table: hospital_metadata_full_backup")

# COMMAND ----------

# Cell 3 — Fix regions + remove duplicates
import pandas as pd

df = spark.table("hospital_metadata_full").toPandas().fillna("")
print(f"Starting: {len(df)} rows, {df['name'].nunique()} unique hospitals")

# ── STEP 1: Fix messy region names ──
messy_fixes = {
    'Ga East Municipality, Greater Accra Region': 'Greater Accra',
    'Shai Osudoku District, Greater Accra Region': 'Greater Accra',
    'Ga East Municipality':   'Greater Accra',
    'Accra East':             'Greater Accra',
    'Accra North':            'Greater Accra',
    'Ledzokuku-Krowor':       'Greater Accra',
    'Tema West Municipal':    'Greater Accra',
    'Takoradi':               'Western',
    'Asutifi South':          'Ahafo',
    'Asokwa-Kumasi':          'Ashanti',
    'Ejisu Municipal':        'Ashanti',
    'Ahafo Ano South-East':   'Ashanti',
    'Techiman Municipal':     'Brong Ahafo',
    'Dormaa East':            'Brong Ahafo',
    'Central Tongu District': 'Volta',
    'Sissala West District':  'Upper West',
    'KEEA':                   'Central',
    'Central Ghana':          'Central',
    'Bono':                   'Brong Ahafo',
    'Ghana':                  'Unknown',
    'SH':                     'Unknown',
}
df['region_clean'] = df['region_clean'].replace(messy_fixes)
print(f"✅ Step 1 done: messy regions fixed")

# ── STEP 2: City-based fix for remaining Unknowns ──
city_to_region = {
    'accra': 'Greater Accra', 'tema': 'Greater Accra',
    'madina': 'Greater Accra', 'legon': 'Greater Accra',
    'dansoman': 'Greater Accra', 'lapaz': 'Greater Accra',
    'nungua': 'Greater Accra', 'weija': 'Greater Accra',
    'ashaiman': 'Greater Accra', 'labadi': 'Greater Accra',
    'klagon': 'Greater Accra', 'teshie': 'Greater Accra',
    'osu': 'Greater Accra', 'spintex': 'Greater Accra',
    'tesano': 'Greater Accra', 'ridge': 'Greater Accra',
    'achimota': 'Greater Accra', 'dome': 'Greater Accra',
    'pokuase': 'Greater Accra', 'haatso': 'Greater Accra',
    'ashale-botwe': 'Greater Accra', 'oyarifa': 'Greater Accra',
    'kumasi': 'Ashanti', 'obuasi': 'Ashanti',
    'ejisu': 'Ashanti', 'bekwai': 'Ashanti',
    'konongo': 'Ashanti', 'kokofu': 'Ashanti',
    'suame': 'Ashanti', 'santasi': 'Ashanti',
    'agogo': 'Ashanti', 'kumawu': 'Ashanti',
    'takoradi': 'Western', 'sekondi': 'Western',
    'tarkwa': 'Western', 'axim': 'Western',
    'prestea': 'Western', 'kwesimintsim': 'Western',
    'bibiani': 'Western North', 'sefwi wiawso': 'Western North',
    'juaboso': 'Western North', 'enchi': 'Western North',
    'cape coast': 'Central', 'winneba': 'Central',
    'saltpond': 'Central', 'elmina': 'Central',
    'mankessim': 'Central', 'swedru': 'Central',
    'koforidua': 'Eastern', 'nkawkaw': 'Eastern',
    'suhum': 'Eastern', 'somanya': 'Eastern',
    'ho': 'Volta', 'hohoe': 'Volta',
    'keta': 'Volta', 'aflao': 'Volta',
    'akatsi': 'Volta', 'sogakope': 'Volta',
    'dzodze': 'Volta', 'battor': 'Volta',
    'tamale': 'Northern', 'yendi': 'Northern',
    'savelugu': 'Northern', 'tolon': 'Northern',
    'damongo': 'Savannah', 'bole': 'Savannah',
    'bolgatanga': 'Upper East', 'navrongo': 'Upper East',
    'bawku': 'Upper East',
    'wa': 'Upper West', 'lawra': 'Upper West',
    'tumu': 'Upper West', 'jirapa': 'Upper West',
    'nadawli': 'Upper West', 'daffiama': 'Upper West',
    'sunyani': 'Brong Ahafo', 'techiman': 'Brong Ahafo',
    'berekum': 'Brong Ahafo', 'kintampo': 'Brong Ahafo',
    'bechem': 'Ahafo', 'goaso': 'Ahafo',
    'dambai': 'Oti', 'nkwanta': 'Oti',
    'atebubu': 'Bono East', 'nalerigu': 'North East',
}

fixed_city = 0
def fix_unknown(row):
    global fixed_city
    if row['region_clean'] != 'Unknown':
        return row['region_clean']
    city = str(row.get('address_city', '')).lower().strip()
    for key, region in city_to_region.items():
        if key in city:
            fixed_city += 1
            return region
    text = (str(row.get('name','')) + ' ' + 
            str(row.get('description',''))).lower()
    keywords = {
        'Greater Accra': ['accra','tema','legon'],
        'Ashanti': ['kumasi','ashanti'],
        'Northern': ['tamale','northern ghana'],
        'Upper East': ['bolgatanga','upper east'],
        'Upper West': ['upper west','jirapa'],
        'Western': ['takoradi','sekondi'],
        'Volta': ['hohoe','volta region'],
        'Central': ['cape coast','central region'],
        'Brong Ahafo': ['sunyani','brong ahafo'],
    }
    for region, kws in keywords.items():
        if any(k in text for k in kws):
            fixed_city += 1
            return region
    return 'Unknown'

df['region_clean'] = df.apply(fix_unknown, axis=1)
print(f"✅ Step 2 done: fixed {fixed_city} Unknown regions")

# ── STEP 3: Remove duplicates (keep most complete row) ──
def completeness(row):
    score = 0
    for col in ['description','procedure','equipment',
                'capability','specialties','address_city']:
        if str(row.get(col,'')).strip() not in ['','nan','[]',"['']"]:
            score += 1
    return score

df['_score'] = df.apply(completeness, axis=1)
df = df.sort_values('_score', ascending=False)
df = df.drop_duplicates(subset=['name'], keep='first')
df = df.drop(columns=['_score']).reset_index(drop=True)
print(f"✅ Step 3 done: duplicates removed")

# ── FINAL REPORT ──
print(f"\n=== RESULTS ===")
print(f"Rows before : 987")
print(f"Rows after  : {len(df)}")
print(f"Duplicates removed: {987 - len(df)}")
print(f"Unknown remaining : {(df['region_clean'] == 'Unknown').sum()}")
print(f"\n📊 Clean region distribution:")
counts = df[df['region_clean'] != 'Unknown']['region_clean'].value_counts()
for region, count in counts.items():
    print(f"  {region:<35} → {count}")
print(f"  {'Unknown':<35} → {(df['region_clean'] == 'Unknown').sum()}")

# COMMAND ----------

# Quick check — why is Volta so high?
print("=== VOLTA HOSPITALS SAMPLE ===\n")
volta = df[df['region_clean'] == 'Volta']
print(f"Total Volta hospitals: {len(volta)}")
print(f"\nSample of Volta hospitals:")
for _, row in volta.head(20).iterrows():
    print(f"  {str(row['name'])[:45]:<45} | city: {str(row['address_city']):<20} | orig_region: {str(row.get('address_stateOrRegion',''))[:20]}")

# COMMAND ----------

# Cell 4 — Fix wrong Volta assignments
# The 'ho' keyword was too broad and matched wrong cities

print(f"Volta before fix: {(df['region_clean'] == 'Volta').sum()}")

# Fix hospitals wrongly assigned to Volta
city_corrections = {
    # Should be Greater Accra
    'ashaiman': 'Greater Accra',
    'weija':    'Greater Accra',
    'nungua':   'Greater Accra',
    'pokoase':  'Greater Accra',
    'oyarifa':  'Greater Accra',
    'haatso':   'Greater Accra',
    'labadi':   'Greater Accra',
    'teshie':   'Greater Accra',
    'klagon':   'Greater Accra',
    'dome':     'Greater Accra',
    'pokuase':  'Greater Accra',
    'achimota': 'Greater Accra',
    'spintex':  'Greater Accra',
    'darkuman': 'Greater Accra',
    'abeka':    'Greater Accra',
    'mallam':   'Greater Accra',
    'odorkor':  'Greater Accra',
    'osu':      'Greater Accra',
    'tesano':   'Greater Accra',
    'ridge':    'Greater Accra',

    # Should be Ashanti
    'kumasi':    'Ashanti',
    'kokofu':    'Ashanti',
    'tikrom':    'Ashanti',
    'ejura':     'Ashanti',
    'kumawu':    'Ashanti',
    'offinso':   'Ashanti',
    'suame':     'Ashanti',
    'santasi':   'Ashanti',
    'bantama':   'Ashanti',
    'kwadaso':   'Ashanti',
    'kuntanase': 'Ashanti',
    'bekwai':    'Ashanti',
    'konongo':   'Ashanti',
    'obuasi':    'Ashanti',

    # Should be Brong Ahafo
    'sunyani':        'Brong Ahafo',
    'techiman':       'Brong Ahafo',
    'berekum':        'Brong Ahafo',
    'kintampo':       'Brong Ahafo',
    'dormaa':         'Brong Ahafo',
    'abesim':         'Brong Ahafo',
    'nkoranza':       'Brong Ahafo',
    'wenchi':         'Brong Ahafo',

    # Should be Central
    'mankessim': 'Central',
    'cape coast': 'Central',
    'winneba':   'Central',
    'saltpond':  'Central',
    'elmina':    'Central',
    'swedru':    'Central',
    'kasoa':     'Central',

    # Should be Western
    'takoradi':    'Western',
    'sekondi':     'Western',
    'tarkwa':      'Western',
    'axim':        'Western',
    'prestea':     'Western',
    'kwesimintsim':'Western',

    # Should be Western North
    'sefwi wiawso': 'Western North',
    'bibiani':      'Western North',
    'juaboso':      'Western North',
    'enchi':        'Western North',

    # Should be Eastern
    'koforidua':    'Eastern',
    'nkawkaw':      'Eastern',
    'suhum':        'Eastern',
    'somanya':      'Eastern',
    'donkorkrom':   'Eastern',
    'odonkawkrom':  'Eastern',
    'nsawam':       'Eastern',

    # Should be Upper East
    'bawku':       'Upper East',
    'bolgatanga':  'Upper East',
    'navrongo':    'Upper East',
    'zebilla':     'Upper East',

    # Should be Upper West
    'wa':       'Upper West',
    'lawra':    'Upper West',
    'tumu':     'Upper West',
    'jirapa':   'Upper West',
    'nandom':   'Upper West',
    'daffiama': 'Upper West',
    'nadawli':  'Upper West',

    # Should be Northern
    'tamale':    'Northern',
    'yendi':     'Northern',
    'savelugu':  'Northern',
    'tolon':     'Northern',
    'gushegu':   'Northern',

    # Should be Savannah
    'damongo': 'Savannah',
    'bole':    'Savannah',

    # Should be Oti
    'dambai':  'Oti',
    'nkwanta': 'Oti',

    # Should be Ahafo
    'bechem': 'Ahafo',
    'goaso':  'Ahafo',
    'kukuom': 'Ahafo',
}

corrected = 0

def correct_region(row):
    global corrected
    city = str(row.get('address_city', '')).lower().strip()
    
    # Check city against corrections
    for city_key, correct_region in city_corrections.items():
        if city_key in city:
            if row['region_clean'] != correct_region:
                corrected += 1
                return correct_region
            return row['region_clean']
    
    return row['region_clean']

df['region_clean'] = df.apply(correct_region, axis=1)

print(f"✅ Corrections applied: {corrected} hospitals fixed")
print(f"Volta after fix: {(df['region_clean'] == 'Volta').sum()}")
print(f"\n📊 Updated region distribution:")
counts = df['region_clean'].value_counts()
for region, count in counts.items():
    print(f"  {region:<35} → {count}")

# COMMAND ----------

# Check remaining Volta hospitals
print("=== REMAINING VOLTA HOSPITALS ===\n")
volta = df[df['region_clean'] == 'Volta']
print(f"Total: {len(volta)}\n")
for _, row in volta.iterrows():
    print(f"  {str(row['name'])[:45]:<45} | city: {str(row['address_city']):<20}")

print("\n=== UPPER WEST HOSPITALS SAMPLE ===\n")
uw = df[df['region_clean'] == 'Upper West']
print(f"Total: {len(uw)}\n")
for _, row in uw.iterrows():
    print(f"  {str(row['name'])[:45]:<45} | city: {str(row['address_city']):<20}")

# COMMAND ----------

# Cell 5 — Manual corrections for wrong Volta + Upper West assignments

manual_corrections = {
    # Wrong Volta → correct regions
    'Ropheka Herbal Hospital':                    'Greater Accra',
    'Juaben Government Hospital':                 'Ashanti',
    'New Life Hospital':                          'Greater Accra',
    'The Community Hospital New Ashongman':       'Greater Accra',
    'The Community Hospital Ashongman':           'Greater Accra',
    'Barnor Memorial Hospital':                   'Greater Accra',
    'Karaga District Hospital':                   'Northern',
    'Zabzugu Tatale District Hospital':           'Northern',
    'Apatrapa Hospital, Tepa, Ghana':             'Ashanti',
    'Adiebebahospital - No. 4 Melcom road, Ahodwo,': 'Ashanti',
    'Gwollu Hospital':                            'Upper West',
    'Gwollu Hospital, Sissala West District, Gwo,':  'Upper West',
    'Wassa Akropong Government Hospital':         'Western',
    'Salaga District Hospital':                   'Northern',
    'Nkenkasu District Hospital':                 'Ashanti',
    'New Abirem Government Hospital':             'Eastern',
    'Mankranso Government Hospital':              'Ashanti',
    'Dormaa Presbyterian Hospital':               'Brong Ahafo',
    'Atebubu Government Hospital':                'Bono East',
    'Phoenix Healthcare':                         'Greater Accra',
    'Carelinx Home Healthcare Agency':            'Greater Accra',
    'National Nuclear Medicine and Radiotherapy Ce': 'Greater Accra',
    'Dompoase':                                   'Greater Accra',
    'Kpendua':                                    'Northern',
    'War Memorial Hospital':                      'Northern',
    'Smile Family Clinic':                        'Northern',
    'Zebilla District Hospital':                  'Upper East',
    'Makango Clinic':                             'Savannah',
    'Planned Parenthood Association of Ghana':    'Upper East',
    'LawFam Nutritional Clinic - 47 Fertilizer Roa': 'Ashanti',
    'Methodist Clinic, Amakom-lake Bosomtwe':     'Ashanti',
    'Mepom RCH':                                  'Ashanti',
    'Pentecost Hospital Ayanfuri':                'Ashanti',
    'Imperial Nursing And Home-care Services':    'Greater Accra',
    'Evergreen Opticals - Burma Camp, Recce Juncti': 'Greater Accra',

    # Wrong Upper West → correct regions
    'SOMACAS Hospital':                           'Oti',
    'Shai-Osudoku District Hospital':             'Greater Accra',
    'Memento Eye Center, Dodowa':                 'Greater Accra',
    'Dangme WEST Hospital Complex':               'Greater Accra',
    'Akwamuman Herbal Clinic':                    'Eastern',
    'Kwabeng RCH':                                'Eastern',
    'St. Dominic Catholic Hospital, Akwatia':     'Eastern',
    'Gloria Memorial Hospital':                   'Eastern',
    'GCD Hospital':                               'Eastern',
    'Ghana consolidated diamond hospital-E/r Akwat': 'Eastern',
    'Saint John of God Hospital, Duayaw Nkwanta': 'Ahafo',
    'St. John of God Catholic Hospital':          'Ahafo',
    'St. Joseph Catholic Hospital, Nkwanta':      'Oti',
    'St.Joseph Hospital, Nkwanta':                'Oti',
    'Princess Town Health Centre':                'Western',
    'APOWA HEALTH CENTRE':                        'Western',
    'Presbyterian Health Centre, Kwame Bikrom':   'Western North',
    'Centre for Plant Medicine Research':         'Eastern',
    'Abuakwa Maternity Home':                     'Ashanti',
    'Gb Dental Clinic (Kwashieman Bus Stop, Kwashi': 'Greater Accra',
    'Dunkwa Municipal Hospital':                  'Central',
    'Al-haq Herbal Center':                       'Northern',
    'Walewale Hospital':                          'North East',
    'Walewale District Hospital':                 'North East',
    'Allen Health Clinic - Ghana':                'Upper West',
    'Woodland Medical Centre':                    'Bono East',
    'Mother- and Child Clinic Twabidi':           'Western North',
    'Jubile Catholic Children Hospital':          'Western',
}

corrected2 = 0
for idx, row in df.iterrows():
    name = str(row['name'])
    for hospital_name, correct_reg in manual_corrections.items():
        if hospital_name in name or name in hospital_name:
            if df.at[idx, 'region_clean'] != correct_reg:
                df.at[idx, 'region_clean'] = correct_reg
                corrected2 += 1
            break

print(f"✅ Manual corrections applied: {corrected2}")
print(f"\n📊 Final region distribution:")
counts = df['region_clean'].value_counts()
for region, count in counts.items():
    print(f"  {region:<35} → {count}")
print(f"\nTotal hospitals: {len(df)}")
print(f"Unknown remaining: {(df['region_clean'] == 'Unknown').sum()}")

# COMMAND ----------

# Cell 6 — Save fixed data permanently
print("🔄 Saving to Delta table...")

spark_df = spark.createDataFrame(
    df.astype(str).fillna('')
)

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_full")

# Verify
saved = spark.table("hospital_metadata_full").toPandas()
unknown = (saved['region_clean'] == 'Unknown').sum()

print(f"✅ Saved successfully!")
print(f"   Rows saved      : {len(saved)}")
print(f"   Unknown regions : {unknown}")
print(f"\n✅ Genie will now give accurate answers!")
print(f"✅ RAG search will now use correct regions!")

# COMMAND ----------

# MAGIC %pip install groq sentence-transformers pydantic -q

# COMMAND ----------

import os, json, time
import pandas as pd
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow

GROQ_KEY = os.environ.get("GROQ_KEY")
client = Groq(api_key=GROQ_KEY)

# Load your existing enriched data
df = spark.table("enriched_facilities").toPandas().fillna("")
print(f"✅ Loaded {len(df)} hospitals")
print(f"   Has description: {(df['description'] != '').sum()}")

# COMMAND ----------

# Sync region fixes from hospital_metadata_full → enriched_facilities

print("🔄 Loading both tables...")

# Load the FIXED master table
master = spark.table("hospital_metadata_full").toPandas().fillna("")

# Load enriched_facilities (has old dirty regions)
enriched = spark.table("enriched_facilities").toPandas().fillna("")

print(f"   master rows    : {len(master)}")
print(f"   enriched rows  : {len(enriched)}")
print(f"   enriched Unknown before: {(enriched['region_clean'] == 'Unknown').sum()}")

# Build a name → region lookup from the FIXED master table
region_lookup = dict(zip(master['name'], master['region_clean']))

# Apply the fixed regions to enriched_facilities
def fix_region(row):
    fixed = region_lookup.get(row['name'], None)
    if fixed and fixed != 'Unknown':
        return fixed
    return row.get('region_clean', 'Unknown')

enriched['region_clean'] = enriched.apply(fix_region, axis=1)

print(f"   enriched Unknown after : {(enriched['region_clean'] == 'Unknown').sum()}")
print(f"\n📊 enriched_facilities region distribution:")
counts = enriched['region_clean'].value_counts()
for region, count in counts.items():
    print(f"  {region:<35} → {count}")

# Save back
spark.createDataFrame(enriched.astype(str).fillna('')) \
     .write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("enriched_facilities")

print(f"\n✅ enriched_facilities synced and saved!")
print(f"   Total rows: {len(enriched)}")

# COMMAND ----------

# Fix the 1 remaining "Ghana" row + verify everything is clean

df_e = spark.table("enriched_facilities").toPandas().fillna("")

# Fix "Ghana" → Unknown
df_e.loc[df_e['region_clean'] == 'Ghana', 'region_clean'] = 'Unknown'

# Final counts
print("✅ Final enriched_facilities status:")
print(f"   Total rows     : {len(df_e)}")
print(f"   Known regions  : {(df_e['region_clean'] != 'Unknown').sum()}")
print(f"   Unknown        : {(df_e['region_clean'] == 'Unknown').sum()}")
print(f"   'Ghana' rows   : {(df_e['region_clean'] == 'Ghana').sum()}")

# Save
spark.createDataFrame(df_e.astype(str).fillna('')) \
     .write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("enriched_facilities")

print("\n✅ Clean! Ready for upgraded extraction.")

# COMMAND ----------

# MAGIC %pip install groq pydantic -q

# COMMAND ----------

import os, json, time
import pandas as pd
from groq import Groq
import mlflow

GROQ_KEY = os.environ.get("GROQ_KEY")
client = Groq(api_key=GROQ_KEY)

df = spark.table("enriched_facilities").toPandas().fillna("")
print(f"✅ Loaded {len(df)} hospitals")
print(f"   Has description: {(df['description'] != '').sum()}")

# COMMAND ----------

# Official Virtue Foundation Pydantic Models + Specialties

VALID_SPECIALTIES = [
    "internalMedicine", "familyMedicine", "emergencyMedicine", "pediatrics",
    "gynecologyAndObstetrics", "generalSurgery", "cardiology", "cardiacSurgery",
    "orthopedicSurgery", "ophthalmology", "otolaryngology", "dentistry",
    "radiology", "pathology", "anesthesia", "criticalCareMedicine",
    "infectiousDiseases", "medicalOncology", "nephrology",
    "physicalMedicineAndRehabilitation", "geriatricsInternalMedicine",
    "endocrinologyAndDiabetesAndMetabolism", "neonatologyPerinatalMedicine",
    "hospiceAndPalliativeInternalMedicine", "plasticSurgery", "orthodontics",
    "neurology", "psychiatry", "dermatology", "urology", "gastroenterology",
]

def extract_official(hospital_name, description, facility_type, client, retries=2):
    """
    Uses official Virtue Foundation prompt structure.
    Extracts procedure + equipment + capability + specialties + citations.
    """
    if len(str(description).strip()) < 30:
        return {
            "procedure": [], "equipment": [], "capability": [],
            "specialties": [], "citations": {}, "status": "SKIPPED_SHORT"
        }

    system_prompt = f"""You are a specialized medical facility information extractor for the Virtue Foundation.
Extract structured facts ONLY about this organization: `{hospital_name}`

CATEGORY DEFINITIONS:
- procedure: Clinical procedures, surgical operations, medical interventions, diagnostic procedures 
  and screenings performed at this facility. Clear declarative statements.
- equipment: Physical medical devices and infrastructure — imaging machines (MRI/CT/X-ray), 
  surgical/OR technologies, lab analyzers, oxygen plants, backup power. Include specific models.
- capability: Medical capabilities defining care level — trauma/emergency care levels, specialized 
  units (ICU/NICU/burn unit), clinical programs, accreditations, inpatient/outpatient, staffing, 
  patient capacity. Excludes addresses, contact info, business hours, pricing.

SPECIALTY RULES (use exact case-sensitive values only):
- Generic hospital → internalMedicine
- Generic clinic → familyMedicine
- Emergency/ER → emergencyMedicine
- Maternity/Obstetric/Gynecology → gynecologyAndObstetrics
- Surgery/Surgical → generalSurgery
- Pediatric/Children → pediatrics
- Dental → dentistry
- Eye/Ophthalmic → ophthalmology
- Cardiac Surgery → cardiacSurgery
- Cardiology (non-surgical) → cardiology
- ICU/Critical care → criticalCareMedicine
- Facility type is: {facility_type}

CITATION RULE:
For each fact, capture the exact short phrase from the description that supports it.

Return ONLY this JSON, no other text:
{{
    "procedure": ["declarative fact 1", "declarative fact 2"],
    "equipment": ["declarative fact 1", "declarative fact 2"],
    "capability": ["declarative fact 1", "declarative fact 2"],
    "specialties": ["specialty1", "specialty2"],
    "citations": {{
        "procedure": ["source phrase 1", "source phrase 2"],
        "equipment": ["source phrase 1", "source phrase 2"],
        "capability": ["source phrase 1", "source phrase 2"]
    }}
}}"""

    user_prompt = f"""Extract all medical facts for: {hospital_name}

Description:
{str(description)[:1500]}

Return ONLY valid JSON."""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                n=1,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            result = json.loads(raw)

            # Validate specialties
            valid_specs = [s for s in result.get("specialties", []) 
                          if s in VALID_SPECIALTIES]

            return {
                "procedure":   result.get("procedure", []),
                "equipment":   result.get("equipment", []),
                "capability":  result.get("capability", []),
                "specialties": valid_specs,
                "citations":   result.get("citations", {}),
                "status":      "SUCCESS"
            }

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return {"procedure":[],"equipment":[],"capability":[],
                    "specialties":[],"citations":{},"status":"JSON_ERROR"}
        except Exception as e:
            err = str(e)
            if 'rate_limit' in err.lower():
                wait = (attempt + 1) * 15
                print(f"     ⏳ Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                return {"procedure":[],"equipment":[],"capability":[],
                        "specialties":[],"citations":{},"status":f"ERROR:{err[:40]}"}

    return {"procedure":[],"equipment":[],"capability":[],
            "specialties":[],"citations":{},"status":"FAILED"}

print(f"✅ Official extraction function ready!")
print(f"   Specialties list: {len(VALID_SPECIALTIES)} valid values")

# COMMAND ----------

# Test on 3 hospitals with rich descriptions before running all 987

test_df = df[df['description'].str.len() > 200].head(3)

print("🧪 TESTING ON 3 HOSPITALS\n")
print("=" * 65)

for i, row in test_df.iterrows():
    print(f"\n🏥 {row['name']}")
    print(f"📝 {str(row['description'])[:120]}...")

    result = extract_official(
        row['name'],
        row['description'],
        str(row.get('facilityTypeId', 'hospital')),
        client
    )

    print(f"\n   Status      : {result['status']}")
    print(f"   Procedures  : {result['procedure'][:2]}")
    print(f"   Equipment   : {result['equipment'][:2]}")
    print(f"   Capabilities: {result['capability'][:2]}")
    print(f"   Specialties : {result['specialties']}")
    print(f"   Citations   : {list(result['citations'].keys())}")
    print("-" * 65)
    time.sleep(3)

print("\n✅ Test done! If results look good, run Cell 5 for all 987.")

# COMMAND ----------

# Cell 5 — Process all 987 hospitals with official extraction

print("🚀 Starting official extraction on all 987 hospitals...")
print("   Estimated time: ~25-30 minutes (rate limit delays)")
print("   Progress updates every 50 hospitals\n")

results_list = []
success = 0
skipped = 0
errors  = 0

for i, row in df.iterrows():
    idx = len(results_list)

    # Progress update
    if idx % 50 == 0 and idx > 0:
        print(f"   [{idx}/987] ✅ {success} success | "
              f"⏭️ {skipped} skipped | ❌ {errors} errors")

    result = extract_official(
        str(row['name']),
        str(row['description']),
        str(row.get('facilityTypeId', 'hospital')),
        client
    )

    # Track counts
    if result['status'] == 'SUCCESS':
        success += 1
    elif result['status'] == 'SKIPPED_SHORT':
        skipped += 1
    else:
        errors += 1

    results_list.append({
        'name':        row['name'],
        'region_clean': row['region_clean'],
        'procedure_v2':  json.dumps(result['procedure']),
        'equipment_v2':  json.dumps(result['equipment']),
        'capability_v2': json.dumps(result['capability']),
        'specialties_v2': json.dumps(result['specialties']),
        'citations_v2':   json.dumps(result['citations']),
        'extract_status': result['status'],
    })

    # Rate limit protection — 1.5s between calls
    time.sleep(1.5)

print(f"\n{'='*50}")
print(f"✅ EXTRACTION COMPLETE!")
print(f"   Success  : {success}")
print(f"   Skipped  : {skipped}")
print(f"   Errors   : {errors}")
print(f"   Total    : {len(results_list)}")
print(f"{'='*50}")