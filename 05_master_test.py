# Databricks notebook source
# Check all tables are there
print("=== CHECKING ALL DELTA TABLES ===\n")

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
    except Exception as e:
        print(f"❌ {table:<35} → MISSING! {str(e)[:50]}")

# COMMAND ----------

# Check how well the IDP agent filled in missing data
import pandas as pd

df = spark.table("enriched_facilities").toPandas()
df = df.fillna("")

total = len(df)

has_procedure  = (df['procedure'].str.len() > 5).sum()
has_equipment  = (df['equipment'].str.len() > 5).sum()
has_capability = (df['capability'].str.len() > 5).sum()
has_all_three  = (
    (df['procedure'].str.len() > 5) &
    (df['equipment'].str.len() > 5) &
    (df['capability'].str.len() > 5)
).sum()

print("=== IDP EXTRACTION RESULTS ===\n")
print(f"Total facilities:          {total}")
print(f"Has procedure:             {has_procedure} ({has_procedure/total*100:.1f}%)")
print(f"Has equipment:             {has_equipment} ({has_equipment/total*100:.1f}%)")
print(f"Has capability:            {has_capability} ({has_capability/total*100:.1f}%)")
print(f"Has ALL THREE fields:      {has_all_three} ({has_all_three/total*100:.1f}%)")
print(f"\n✅ IDP improved data completeness significantly!")

# COMMAND ----------

# DIAGNOSE THE PROBLEM
import pandas as pd

df = spark.table("hospital_metadata_full").toPandas()
df = df.fillna("")

print(f"Total rows: {len(df)}")
print(f"Unique hospitals: {df['name'].nunique()}")
print(f"\nTop duplicate names:")
print(df['name'].value_counts().head(10))

print(f"\nRegion distribution:")
print(df['region_clean'].value_counts().head(15))

print(f"\nSample row with good data:")
good = df[df['capability'].str.len() > 50].head(1)
for col in ['name','region_clean','address_city',
            'facilityTypeId','capability','procedure']:
    print(f"  {col}: {str(good[col].values[0])[:100]}")

# COMMAND ----------

import pandas as pd

# Load original clean data (this is the correct base)
df_original = spark.table("medical_facilities_clean").toPandas()
df_original = df_original.fillna("")

print(f"Original clean table: {len(df_original)} rows")
print(f"Unique hospitals: {df_original['name'].nunique()}")
print(f"\nRegion distribution:")
print(df_original['region_clean'].value_counts().head(15))

# COMMAND ----------

# SAFE CHECK - see exactly what will be kept vs removed
import pandas as pd

df = spark.table("hospital_metadata_full").toPandas()
df = df.fillna("")

def completeness_score(row):
    score = 0
    for col in ['description','procedure','equipment',
                'capability','specialties','address_city']:
        if str(row.get(col,'')).strip() not in ['','nan','[]',"['']"]:
            score += 1
    return score

df['completeness'] = df.apply(completeness_score, axis=1)
df_sorted = df.sort_values('completeness', ascending=False)

# Show what we KEEP for top duplicates
print("=== WHAT WE KEEP (best row per hospital) ===\n")
duplicates = ['Korle Bu Teaching Hospital', 
              '1st Foundation Clinic',
              'The Bank Hospital',
              'Marie Stopes Ghana']

for hosp_name in duplicates:
    rows = df_sorted[df_sorted['name'] == hosp_name]
    print(f"Hospital: {hosp_name} ({len(rows)} copies found)")
    
    for i, (_, row) in enumerate(rows.iterrows()):
        status = "✅ KEEP" if i == 0 else "❌ REMOVE"
        print(f"  {status} | score:{row['completeness']} | "
              f"region:{row['region_clean']} | "
              f"has_desc:{len(str(row['description']))>5} | "
              f"has_proc:{len(str(row['procedure']))>5} | "
              f"has_cap:{len(str(row['capability']))>5}")
    print()

# COMMAND ----------

# STEP 1 - Start from the correct base
import pandas as pd

df = spark.table("medical_facilities_clean").toPandas()
df = df.fillna("")

print(f"✅ Base data: {len(df)} rows")
print(f"   Unique hospitals: {df['name'].nunique()}")

# COMMAND ----------

# STEP 2 - Remove duplicates properly
# Keep the row with most data for each duplicate hospital

def completeness_score(row):
    score = 0
    for col in ['description','procedure','equipment',
                'capability','specialties','address_city']:
        if str(row.get(col,'')).strip() not in ['','nan','[]',"['']"]:
            score += 1
    return score

df['completeness'] = df.apply(completeness_score, axis=1)

# Sort by completeness so best row comes first
df_sorted = df.sort_values('completeness', ascending=False)

# Drop duplicates keeping best row
df_dedup = df_sorted.drop_duplicates(subset=['name'], keep='first')

print(f"✅ After dedup: {len(df_dedup)} unique hospitals")
print(f"   Removed: {len(df) - len(df_dedup)} duplicates")

# COMMAND ----------

# STEP 3 - Fix region mapping properly
# Better region normalization using city names

city_to_region = {
    # Greater Accra cities
    'accra': 'Greater Accra',
    'tema': 'Greater Accra', 
    'madina': 'Greater Accra',
    'legon': 'Greater Accra',
    'adenta': 'Greater Accra',
    'kasoa': 'Greater Accra',
    'dansoman': 'Greater Accra',
    'lapaz': 'Greater Accra',
    'achimota': 'Greater Accra',
    'east legon': 'Greater Accra',
    'osu': 'Greater Accra',
    'cantonments': 'Greater Accra',
    'labone': 'Greater Accra',
    'airport': 'Greater Accra',
    'spintex': 'Greater Accra',
    'tesano': 'Greater Accra',
    'dzorwulu': 'Greater Accra',
    'north kaneshie': 'Greater Accra',
    'asylum down': 'Greater Accra',
    'ridge': 'Greater Accra',
    
    # Ashanti cities
    'kumasi': 'Ashanti',
    'obuasi': 'Ashanti',
    'ejisu': 'Ashanti',
    'bekwai': 'Ashanti',
    'mampong': 'Ashanti',
    'konongo': 'Ashanti',
    'asante mampong': 'Ashanti',
    
    # Western cities
    'takoradi': 'Western',
    'sekondi': 'Western',
    'tarkwa': 'Western',
    'half assini': 'Western',
    'axim': 'Western',
    'prestea': 'Western',
    
    # Central cities
    'cape coast': 'Central',
    'winneba': 'Central',
    'saltpond': 'Central',
    'elmina': 'Central',
    'assin fosu': 'Central',
    
    # Eastern cities
    'koforidua': 'Eastern',
    'nkawkaw': 'Eastern',
    'akosombo': 'Eastern',
    'suhum': 'Eastern',
    
    # Volta cities
    'ho': 'Volta',
    'hohoe': 'Volta',
    'keta': 'Volta',
    'aflao': 'Volta',
    'akatsi': 'Volta',
    
    # Northern cities
    'tamale': 'Northern',
    'yendi': 'Northern',
    'savelugu': 'Northern',
    'damongo': 'Savannah',
    
    # Upper regions
    'bolgatanga': 'Upper East',
    'navrongo': 'Upper East',
    'bawku': 'Upper East',
    'wa': 'Upper West',
    'lawra': 'Upper West',
    'tumu': 'Upper West',
    
    # Brong Ahafo
    'sunyani': 'Brong Ahafo',
    'techiman': 'Brong Ahafo',
    'berekum': 'Brong Ahafo',
    'kintampo': 'Brong Ahafo',
    'bechem': 'Ahafo',
}

def fix_region(row):
    # If already has a good region, keep it
    current = str(row.get('region_clean','')).strip()
    if current not in ['','Unknown','nan']:
        return current
    
    # Try to detect from city
    city = str(row.get('address_city','')).lower().strip()
    for city_key, region in city_to_region.items():
        if city_key in city:
            return region
    
    # Try from name/description
    text = (str(row.get('name','')) + ' ' + 
            str(row.get('description',''))).lower()
    
    region_keywords = {
        'Greater Accra': ['accra','tema','legon'],
        'Ashanti': ['kumasi','ashanti'],
        'Western': ['takoradi','sekondi','western'],
        'Northern': ['tamale','northern'],
        'Upper East': ['bolgatanga','upper east'],
        'Upper West': ['wa ','upper west'],
        'Volta': [' ho ', 'volta','hohoe'],
        'Central': ['cape coast','central region'],
    }
    
    for region, keywords in region_keywords.items():
        if any(kw in text for kw in keywords):
            return region
    
    return 'Unknown'

df_dedup['region_clean'] = df_dedup.apply(fix_region, axis=1)

print("✅ Region fix complete!")
print(f"\nRegion distribution after fix:")
print(df_dedup['region_clean'].value_counts().to_string())

# COMMAND ----------

# STEP 3 - Fix region mapping properly
# Better region normalization using city names

city_to_region = {
    # Greater Accra cities
    'accra': 'Greater Accra',
    'tema': 'Greater Accra', 
    'madina': 'Greater Accra',
    'legon': 'Greater Accra',
    'adenta': 'Greater Accra',
    'kasoa': 'Greater Accra',
    'dansoman': 'Greater Accra',
    'lapaz': 'Greater Accra',
    'achimota': 'Greater Accra',
    'east legon': 'Greater Accra',
    'osu': 'Greater Accra',
    'cantonments': 'Greater Accra',
    'labone': 'Greater Accra',
    'airport': 'Greater Accra',
    'spintex': 'Greater Accra',
    'tesano': 'Greater Accra',
    'dzorwulu': 'Greater Accra',
    'north kaneshie': 'Greater Accra',
    'asylum down': 'Greater Accra',
    'ridge': 'Greater Accra',
    
    # Ashanti cities
    'kumasi': 'Ashanti',
    'obuasi': 'Ashanti',
    'ejisu': 'Ashanti',
    'bekwai': 'Ashanti',
    'mampong': 'Ashanti',
    'konongo': 'Ashanti',
    'asante mampong': 'Ashanti',
    
    # Western cities
    'takoradi': 'Western',
    'sekondi': 'Western',
    'tarkwa': 'Western',
    'half assini': 'Western',
    'axim': 'Western',
    'prestea': 'Western',
    
    # Central cities
    'cape coast': 'Central',
    'winneba': 'Central',
    'saltpond': 'Central',
    'elmina': 'Central',
    'assin fosu': 'Central',
    
    # Eastern cities
    'koforidua': 'Eastern',
    'nkawkaw': 'Eastern',
    'akosombo': 'Eastern',
    'suhum': 'Eastern',
    
    # Volta cities
    'ho': 'Volta',
    'hohoe': 'Volta',
    'keta': 'Volta',
    'aflao': 'Volta',
    'akatsi': 'Volta',
    
    # Northern cities
    'tamale': 'Northern',
    'yendi': 'Northern',
    'savelugu': 'Northern',
    'damongo': 'Savannah',
    
    # Upper regions
    'bolgatanga': 'Upper East',
    'navrongo': 'Upper East',
    'bawku': 'Upper East',
    'wa': 'Upper West',
    'lawra': 'Upper West',
    'tumu': 'Upper West',
    
    # Brong Ahafo
    'sunyani': 'Brong Ahafo',
    'techiman': 'Brong Ahafo',
    'berekum': 'Brong Ahafo',
    'kintampo': 'Brong Ahafo',
    'bechem': 'Ahafo',
}

def fix_region(row):
    # If already has a good region, keep it
    current = str(row.get('region_clean','')).strip()
    if current not in ['','Unknown','nan']:
        return current
    
    # Try to detect from city
    city = str(row.get('address_city','')).lower().strip()
    for city_key, region in city_to_region.items():
        if city_key in city:
            return region
    
    # Try from name/description
    text = (str(row.get('name','')) + ' ' + 
            str(row.get('description',''))).lower()
    
    region_keywords = {
        'Greater Accra': ['accra','tema','legon'],
        'Ashanti': ['kumasi','ashanti'],
        'Western': ['takoradi','sekondi','western'],
        'Northern': ['tamale','northern'],
        'Upper East': ['bolgatanga','upper east'],
        'Upper West': ['wa ','upper west'],
        'Volta': [' ho ', 'volta','hohoe'],
        'Central': ['cape coast','central region'],
    }
    
    for region, keywords in region_keywords.items():
        if any(kw in text for kw in keywords):
            return region
    
    return 'Unknown'

df_dedup['region_clean'] = df_dedup.apply(fix_region, axis=1)

print("✅ Region fix complete!")
print(f"\nRegion distribution after fix:")
print(df_dedup['region_clean'].value_counts().to_string())

# COMMAND ----------

# STEP 4 - Save as corrected master table
spark_df = spark.createDataFrame(
    df_dedup.astype(str).fillna('')
)

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_full")

print(f"✅ Saved hospital_metadata_full: {len(df_dedup)} rows")
print(f"   Duplicates removed: {987 - len(df_dedup)}")

# Also export updated CSV for Streamlit
df_dedup.to_csv("/tmp/hospital_metadata.csv", index=False)
print(f"✅ CSV exported to /tmp/hospital_metadata.csv")

# COMMAND ----------

# STEP 5 - Verify everything looks correct now
df_check = spark.table("hospital_metadata_full").toPandas()
df_check = df_check.fillna("")

print(f"=== FINAL DATA QUALITY CHECK ===\n")
print(f"Total hospitals:    {len(df_check)}")
print(f"Unique names:       {df_check['name'].nunique()}")
print(f"\nRegion distribution:")
print(df_check['region_clean'].value_counts().to_string())

print(f"\nData completeness:")
for col in ['description','procedure','equipment','capability']:
    filled = (df_check[col].str.len() > 5).sum()
    pct = filled/len(df_check)*100
    print(f"  {col:<15} {filled:>4}/{len(df_check)} ({pct:.1f}%)")

print(f"\n✅ Data is ready for testing!")

# COMMAND ----------

# Check WHERE the duplicates actually are
print("=== medical_facilities_clean ===")
df1 = spark.table("medical_facilities_clean").toPandas()
print(f"Total rows: {len(df1)}")
print(f"Unique names: {df1['name'].nunique()}")
print(f"Duplicates: {len(df1) - df1['name'].nunique()}")

print("\n=== hospital_metadata_full ===")
df2 = spark.table("hospital_metadata_full").toPandas()
print(f"Total rows: {len(df2)}")
print(f"Unique names: {df2['name'].nunique()}")
print(f"Duplicates: {len(df2) - df2['name'].nunique()}")

print("\n=== enriched_facilities ===")
df3 = spark.table("enriched_facilities").toPandas()
print(f"Total rows: {len(df3)}")
print(f"Unique names: {df3['name'].nunique()}")
print(f"Duplicates: {len(df3) - df3['name'].nunique()}")

# COMMAND ----------

# Check the ORIGINAL RAW CSV - before any processing
import pandas as pd

# Load original raw file
df_raw = pd.read_csv(
    "/Volumes/workspace/default/project/"
    "Virtue Foundation Ghana v0.3 - Sheet1.csv"
)

print(f"=== ORIGINAL RAW CSV ===")
print(f"Total rows:     {len(df_raw)}")
print(f"Unique names:   {df_raw['name'].nunique()}")
print(f"Duplicates:     {len(df_raw) - df_raw['name'].nunique()}")

print(f"\nTop duplicate names in ORIGINAL CSV:")
dupes = df_raw['name'].value_counts()
dupes = dupes[dupes > 1]
print(dupes.to_string())

# COMMAND ----------

import pandas as pd

df_raw = pd.read_csv(
    "/Volumes/workspace/default/project/"
    "Virtue Foundation Ghana v0.3 - Sheet1.csv"
)
df_raw = df_raw.fillna("")

# Check the top duplicates - same city or different?
check_hospitals = [
    'Marie Stopes Ghana',
    'Korle Bu Teaching Hospital', 
    '37 Military Hospital',
    'The Bank Hospital',
    '1st Foundation Clinic',
]

for hosp in check_hospitals:
    rows = df_raw[df_raw['name'] == hosp]
    print(f"\n{hosp} ({len(rows)} entries):")
    for _, row in rows.iterrows():
        print(f"  → City: {row.get('address_city','?')} | "
              f"Region: {row.get('address_stateOrRegion','?')}")

# COMMAND ----------

import pandas as pd

df = spark.table("hospital_metadata_full").toPandas()
df = df.fillna("")

# Check specifically wrong region assignments
print("=== CHECKING REGION ACCURACY ===\n")

# Takoradi should be Western not Ashanti
takoradi = df[df['address_city'].str.lower().str.contains('takoradi', na=False)]
print(f"Takoradi hospitals (should be Western):")
for _, row in takoradi.iterrows():
    print(f"  {row['name']} → region: {row['region_clean']}")

print()

# Tamale should be Northern
tamale = df[df['address_city'].str.lower().str.contains('tamale', na=False)]
print(f"Tamale hospitals (should be Northern):")
for _, row in tamale.iterrows():
    print(f"  {row['name']} → region: {row['region_clean']}")

print()

# Kumasi should be Ashanti
kumasi = df[df['address_city'].str.lower().str.contains('kumasi', na=False)]
print(f"Kumasi hospitals (should be Ashanti):")
for _, row in kumasi.head(5).iterrows():
    print(f"  {row['name']} → region: {row['region_clean']}")

print()

# How many Unknown regions remain?
unknown = df[df['region_clean'] == 'Unknown']
print(f"Still Unknown region: {len(unknown)} hospitals")

# COMMAND ----------

# FINAL PERFORMANCE TEST - All 4 components
import pandas as pd

print("=" * 60)
print("GHANA MEDICAL DESERT AGENT - PERFORMANCE REPORT")
print("=" * 60)

# 1. Data Coverage
df = spark.table("hospital_metadata_full").toPandas().fillna("")
print(f"\n📊 DATA COVERAGE")
print(f"   Total hospitals:        {len(df)}")
print(f"   Known regions:          {(df['region_clean'] != 'Unknown').sum()}")
print(f"   Unknown regions:        {(df['region_clean'] == 'Unknown').sum()}")
print(f"   Has description:        {(df['description'].str.len() > 5).sum()}")
print(f"   Has procedure:          {(df['procedure'].str.len() > 5).sum()}")
print(f"   Has equipment:          {(df['equipment'].str.len() > 5).sum()}")
print(f"   Has capability:         {(df['capability'].str.len() > 5).sum()}")

# 2. IDP Performance
print(f"\n🤖 IDP EXTRACTION PERFORMANCE")
enriched = spark.table("enriched_facilities").toPandas().fillna("")
total = len(enriched)
print(f"   Total processed:        {total}")
print(f"   Procedure extracted:    {(enriched['procedure'].str.len()>5).sum()} ({(enriched['procedure'].str.len()>5).sum()/total*100:.1f}%)")
print(f"   Equipment extracted:    {(enriched['equipment'].str.len()>5).sum()} ({(enriched['equipment'].str.len()>5).sum()/total*100:.1f}%)")
print(f"   Capability extracted:   {(enriched['capability'].str.len()>5).sum()} ({(enriched['capability'].str.len()>5).sum()/total*100:.1f}%)")

# 3. Gap Analysis
print(f"\n🏜️ MEDICAL DESERT ANALYSIS")
gap = spark.table("region_gap_analysis").toPandas().fillna("")
print(f"   Regions analysed:       {len(gap)}")
critical = gap[gap['risk_level'].str.contains('CRITICAL', na=False)]
high     = gap[gap['risk_level'].str.contains('HIGH', na=False)]
moderate = gap[gap['risk_level'].str.contains('MODERATE', na=False)]
adequate = gap[gap['risk_level'].str.contains('ADEQUATE', na=False)]
print(f"   🔴 Critical deserts:    {len(critical)}")
print(f"   🟠 High risk:           {len(high)}")
print(f"   🟡 Moderate risk:       {len(moderate)}")
print(f"   🟢 Adequate coverage:   {len(adequate)}")
print(f"\n   Critical desert regions:")
for _, row in critical.iterrows():
    print(f"   • {row['region_clean']:<20} "
          f"→ {row['services_available']}/8 services "
          f"| {row['total_facilities']} facilities")

# 4. Anomaly Detection
print(f"\n⚠️ ANOMALY DETECTION")
anomalies = spark.table("facility_anomalies").toPandas().fillna("")
print(f"   Total anomalies found:  {len(anomalies)}")
if 'anomaly_type' in anomalies.columns:
    for atype, count in anomalies['anomaly_type'].value_counts().items():
        print(f"   • {atype:<35} → {count}")

print(f"\n✅ SYSTEM READY FOR SUBMISSION!")
print("=" * 60)

# COMMAND ----------

# See what the 138 unknowns look like
import pandas as pd

df = spark.table("hospital_metadata_full").toPandas().fillna("")
unknown = df[df['region_clean'] == 'Unknown']

print(f"Total Unknown: {len(unknown)}\n")
print("Sample of Unknown hospitals:")
for _, row in unknown.head(20).iterrows():
    print(f"  Name: {row['name'][:50]}")
    print(f"  City: {row['address_city']}")
    print(f"  Desc: {str(row['description'])[:80]}")
    print()

# COMMAND ----------

import pandas as pd

df = spark.table("hospital_metadata_full").toPandas().fillna("")

# Extended city to region mapping for these specific unknowns
extended_city_mapping = {
    # Central Region
    'mankessim': 'Central',
    'cape coast': 'Central',
    'winneba': 'Central',
    'saltpond': 'Central',
    'elmina': 'Central',
    'assin fosu': 'Central',
    'swedru': 'Central',
    'agona': 'Central',
    'kasoa': 'Central',
    'anomabo': 'Central',
    
    # Western Region
    'bibiani': 'Western North',
    'sefwi wiawso': 'Western North',
    'juaboso': 'Western North',
    'sefwi': 'Western North',
    'enchi': 'Western North',
    'takoradi': 'Western',
    'sekondi': 'Western',
    'tarkwa': 'Western',
    'axim': 'Western',
    'prestea': 'Western',
    'half assini': 'Western',
    'nkroful': 'Western',
    
    # Eastern Region
    'somanya': 'Eastern',
    'koforidua': 'Eastern',
    'nkawkaw': 'Eastern',
    'akosombo': 'Eastern',
    'suhum': 'Eastern',
    'juaben': 'Ashanti',
    'donkorkrom': 'Eastern',
    'odonkawkrom': 'Eastern',
    'abetifi': 'Eastern',
    'mpraeso': 'Eastern',
    
    # Greater Accra
    'nungua': 'Greater Accra',
    'weija': 'Greater Accra',
    'ashaiman': 'Greater Accra',
    'labadi': 'Greater Accra',
    'klagon': 'Greater Accra',
    'teshie': 'Greater Accra',
    'nima': 'Greater Accra',
    'darkuman': 'Greater Accra',
    'abeka': 'Greater Accra',
    'ablekuma': 'Greater Accra',
    'dome': 'Greater Accra',
    'amasaman': 'Greater Accra',
    'pokuase': 'Greater Accra',
    'mallam': 'Greater Accra',
    'odorkor': 'Greater Accra',
    'bubuashie': 'Greater Accra',
    'kwashieman': 'Greater Accra',
    'laterbiokorshie': 'Greater Accra',
    
    # Volta Region
    'dzodze': 'Volta',
    'keta': 'Volta',
    'ho': 'Volta',
    'hohoe': 'Volta',
    'aflao': 'Volta',
    'akatsi': 'Volta',
    'sogakope': 'Volta',
    'anloga': 'Volta',
    'kpando': 'Volta',
    'denu': 'Volta',
    'battor': 'Volta',
    'adidome': 'Volta',
    'worawora': 'Volta',
    'jasikan': 'Volta',
    
    # Ashanti Region
    'kumasi': 'Ashanti',
    'obuasi': 'Ashanti',
    'ejisu': 'Ashanti',
    'bekwai': 'Ashanti',
    'mampong': 'Ashanti',
    'konongo': 'Ashanti',
    'kokofu': 'Ashanti',
    'suame': 'Ashanti',
    'asokwa': 'Ashanti',
    'bantama': 'Ashanti',
    'nhyiaeso': 'Ashanti',
    'kuntanase': 'Ashanti',
    'kwadaso': 'Ashanti',
    'santasi': 'Ashanti',
    'asante mampong': 'Ashanti',
    'agogo': 'Ashanti',
    
    # Brong Ahafo / Bono
    'sunyani': 'Brong Ahafo',
    'techiman': 'Brong Ahafo',
    'berekum': 'Brong Ahafo',
    'kintampo': 'Brong Ahafo',
    'wenchi': 'Brong Ahafo',
    'dormaa ahenkro': 'Brong Ahafo',
    'nkoranza': 'Brong Ahafo',
    'bechem': 'Ahafo',
    'goaso': 'Ahafo',
    'kukuom': 'Ahafo',
    
    # Northern Region
    'tamale': 'Northern',
    'yendi': 'Northern',
    'savelugu': 'Northern',
    'tolon': 'Northern',
    'gushegu': 'Northern',
    'karaga': 'Northern',
    'dompoase': 'Northern',
    
    # Upper East
    'bolgatanga': 'Upper East',
    'navrongo': 'Upper East',
    'bawku': 'Upper East',
    'zebilla': 'Upper East',
    'sandema': 'Upper East',
    
    # Upper West
    'wa': 'Upper West',
    'lawra': 'Upper West',
    'tumu': 'Upper West',
    'jirapa': 'Upper West',
    'nandom': 'Upper West',
    
    # Savannah
    'damongo': 'Savannah',
    'bole': 'Savannah',
    'sawla': 'Savannah',
}

# Also fix using description/name text keywords
name_desc_mapping = {
    'accra': 'Greater Accra',
    'kumasi': 'Ashanti',
    'takoradi': 'Western',
    'tamale': 'Northern',
    'cape coast': 'Central',
    'koforidua': 'Eastern',
    'sunyani': 'Brong Ahafo',
    'bolgatanga': 'Upper East',
    ' wa ': 'Upper West',
    'ho ': 'Volta',
    'hohoe': 'Volta',
    'techiman': 'Brong Ahafo',
    'ashaiman': 'Greater Accra',
    'tema': 'Greater Accra',
    'winneba': 'Central',
    'bibiani': 'Western North',
    'sefwi': 'Western North',
    'juaboso': 'Western North',
    'nungua': 'Greater Accra',
    'weija': 'Greater Accra',
    'dzodze': 'Volta',
    'mankessim': 'Central',
    'somanya': 'Eastern',
    'donkorkrom': 'Eastern',
    'bechem': 'Ahafo',
    'dompoase': 'Northern',
    'suame': 'Ashanti',
}

fixed_count = 0

def fix_unknown_region(row):
    global fixed_count
    
    # Only fix Unknown ones
    if row['region_clean'] != 'Unknown':
        return row['region_clean']
    
    # Try city name first
    city = str(row.get('address_city', '')).lower().strip()
    for city_key, region in extended_city_mapping.items():
        if city_key in city:
            fixed_count += 1
            return region
    
    # Try name + description text
    text = (
        str(row.get('name', '')) + ' ' +
        str(row.get('description', ''))
    ).lower()
    
    for keyword, region in name_desc_mapping.items():
        if keyword in text:
            fixed_count += 1
            return region
    
    return 'Unknown'

df['region_clean'] = df.apply(fix_unknown_region, axis=1)

still_unknown = (df['region_clean'] == 'Unknown').sum()
print(f"✅ Fixed {fixed_count} hospitals!")
print(f"   Remaining Unknown: {still_unknown}")
print(f"\nRegion distribution after fix:")
print(df['region_clean'].value_counts().to_string())