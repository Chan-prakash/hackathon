# Databricks notebook source
# Install Gemini
# %pip install google-genai -q

# COMMAND ----------

# MAGIC # %pip install groq -q

# COMMAND ----------

from groq import Groq
import os

GROQ_KEY = os.environ.get("GROQ_KEY", "")

client = Groq(api_key=GROQ_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # ← Updated model name!
    messages=[
        {"role": "user", "content": "Say exactly: GROQ WORKS PERFECTLY"}
    ]
)

print(response.choices[0].message.content)


# COMMAND ----------

import pandas as pd

# Load the clean data we saved in Notebook 1
df = spark.table("medical_facilities_clean").toPandas()

# Find hospitals that NEED AI extraction
needs_extraction = df[
    df['quality_flags'].str.contains('NEEDS_IDP_EXTRACTION', na=False)
].copy()

print(f"✅ Total facilities: {len(df)}")
print(f"🤖 Needs AI extraction: {len(needs_extraction)}")
print(f"\nSample descriptions to process:")
for i, row in needs_extraction.head(3).iterrows():
    print(f"\n🏥 {row['name']}")
    print(f"   {str(row['description'])[:150]}...")

# COMMAND ----------

import json
import time

def extract_facility_facts(hospital_name, description, client):
    """
    Takes a hospital description
    Returns structured procedure, equipment, capability
    """
    
    prompt = f"""
You are a medical facility information extractor.

Extract facts ONLY about this hospital: {hospital_name}

Description:
{description}

Extract and return a JSON object with exactly these 3 fields:
{{
    "procedure": ["list of medical procedures performed here"],
    "equipment": ["list of physical medical equipment/machines"],
    "capability": ["list of medical capabilities and care levels"]
}}

Rules:
- Only include facts found in the description
- Each item should be a clear declarative sentence
- If nothing found for a category, return empty list []
- Return ONLY the JSON, no other text

Example output:
{{
    "procedure": ["Performs ultrasound scans", "Offers emergency surgery"],
    "equipment": ["Has X-ray machine", "Has operating theatre"],
    "capability": ["Has 24/7 emergency care", "Provides inpatient care"]
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature = more consistent output
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean up response - remove markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        result = json.loads(raw)
        return result
        
    except json.JSONDecodeError:
        return {"procedure": [], "equipment": [], "capability": [], 
                "error": "JSON parse failed"}
    except Exception as e:
        return {"procedure": [], "equipment": [], "capability": [], 
                "error": str(e)}


print("✅ IDP extraction function ready!")

# COMMAND ----------

test_hospitals = needs_extraction.head(3)

print("🧪 TESTING ON 3 HOSPITALS\n")
print("=" * 60)

for i, row in test_hospitals.iterrows():
    print(f"\n🏥 {row['name']}")
    print(f"📝 {str(row['description'])[:150]}...")
    print(f"🤖 Extracting...")
    
    result = extract_facility_facts(
        row['name'],
        row['description'],
        client
    )
    
    print(f"\n✅ RESULTS:")
    print(f"  Procedures : {result.get('procedure', [])}")
    print(f"  Equipment  : {result.get('equipment', [])}")
    print(f"  Capability : {result.get('capability', [])}")
    print("-" * 60)
    time.sleep(1)

# COMMAND ----------

# Find hospitals with LONG detailed descriptions
needs_extraction['desc_length'] = needs_extraction['description'].str.len()

rich_descriptions = needs_extraction[
    needs_extraction['desc_length'] > 300
].sort_values('desc_length', ascending=False)

print(f"Hospitals with rich descriptions: {len(rich_descriptions)}")
print(f"\nTop 5 most detailed descriptions:")
for i, row in rich_descriptions.head(5).iterrows():
    print(f"\n🏥 {row['name']} ({row['desc_length']} chars)")
    print(f"   {str(row['description'])[:200]}...")

# COMMAND ----------

# Test on hospitals with detailed descriptions
test_rich = rich_descriptions.head(5)

print("🧪 TESTING ON DETAILED HOSPITALS\n")
print("=" * 60)

for i, row in test_rich.iterrows():
    print(f"\n🏥 {row['name']}")
    print(f"📝 {str(row['description'])[:200]}...")
    print(f"🤖 Extracting...")
    
    result = extract_facility_facts(
        row['name'],
        row['description'],
        client
    )
    
    print(f"\n✅ RESULTS:")
    print(f"  Procedures ({len(result.get('procedure',[]))}): {result.get('procedure', [])}")
    print(f"  Equipment  ({len(result.get('equipment',[]))}): {result.get('equipment', [])}")
    print(f"  Capability ({len(result.get('capability',[]))}): {result.get('capability', [])}")
    print("-" * 60)
    time.sleep(1)

# COMMAND ----------

# Test on the most detailed hospital first
hospital = rich_descriptions.iloc[0]

print(f"🏥 Hospital: {hospital['name']}")
print(f"📝 Full Description:\n{hospital['description']}")
print(f"\n🤖 Extracting...")

result = extract_facility_facts(
    hospital['name'],
    hospital['description'],
    client
)

print(f"\n✅ EXTRACTION RESULTS:")
print(f"\n📋 Procedures ({len(result.get('procedure',[]))}):")
for p in result.get('procedure', []):
    print(f"   • {p}")
    
print(f"\n🔧 Equipment ({len(result.get('equipment',[]))}):")
for e in result.get('equipment', []):
    print(f"   • {e}")
    
print(f"\n⚡ Capability ({len(result.get('capability',[]))}):")
for c in result.get('capability', []):
    print(f"   • {c}")

# COMMAND ----------

hospital = rich_descriptions.iloc[1]

print(f"🏥 Hospital: {hospital['name']}")
print(f"📝 Full Description:\n{hospital['description']}")
print(f"\n🤖 Extracting...")

result = extract_facility_facts(
    hospital['name'],
    hospital['description'],
    client
)

print(f"\n✅ EXTRACTION RESULTS:")
print(f"\n📋 Procedures ({len(result.get('procedure',[]))}):")
for p in result.get('procedure', []):
    print(f"   • {p}")
    
print(f"\n🔧 Equipment ({len(result.get('equipment',[]))}):")
for e in result.get('equipment', []):
    print(f"   • {e}")
    
print(f"\n⚡ Capability ({len(result.get('capability',[]))}):")
for c in result.get('capability', []):
    print(f"   • {c}")

# COMMAND ----------

def extract_facility_facts_v2(hospital_name, description, client):
    prompt = f"""
You are a medical facility information extractor.
Extract facts ONLY about this hospital: {hospital_name}

Description:
{description}

Extract and return a JSON object with exactly these 3 fields.
Be THOROUGH - extract EVERY medical detail you can find:

{{
    "procedure": [
        "List EVERY medical procedure, treatment, surgery mentioned",
        "Include every department as a procedure",
        "Example: 'Provides dentistry services', 'Performs surgery'"
    ],
    "equipment": [
        "List ALL physical medical equipment and machines",
        "Include labs, pharmacy, radiology as equipment",
        "Example: 'Has laboratory', 'Has radiology equipment'"
    ],
    "capability": [
        "List ALL capabilities - hours, care levels, specialties",
        "Include every department as a capability",
        "Example: 'Has 24/7 emergency care', 'Has pediatrics unit'"
    ]
}}

IMPORTANT RULES:
- Extract EVERY department as both a procedure AND capability
- Extract ALL equipment mentioned
- If you see departments listed, add each one separately
- Return ONLY valid JSON, no other text
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result
        
    except json.JSONDecodeError:
        return {"procedure": [], "equipment": [], "capability": [],
                "error": "JSON parse failed"}
    except Exception as e:
        return {"procedure": [], "equipment": [], "capability": [],
                "error": str(e)}

print("✅ Improved extraction function v2 ready!")

# COMMAND ----------

hospital = rich_descriptions.iloc[1]

print(f"🏥 Hospital: {hospital['name']}")
print(f"🤖 Extracting with improved prompt...")

result = extract_facility_facts_v2(
    hospital['name'],
    hospital['description'],
    client
)

print(f"\n✅ IMPROVED EXTRACTION RESULTS:")
print(f"\n📋 Procedures ({len(result.get('procedure',[]))}):")
for p in result.get('procedure', []):
    print(f"   • {p}")
    
print(f"\n🔧 Equipment ({len(result.get('equipment',[]))}):")
for e in result.get('equipment', []):
    print(f"   • {e}")
    
print(f"\n⚡ Capability ({len(result.get('capability',[]))}):")
for c in result.get('capability', []):
    print(f"   • {c}")

print(f"\n📊 Total facts extracted: {len(result.get('procedure',[])) + len(result.get('equipment',[])) + len(result.get('capability',[]))}")

# COMMAND ----------

# Test on remaining rich hospitals
for idx in [2, 3, 4]:
    hospital = rich_descriptions.iloc[idx]
    
    print(f"\n🏥 {hospital['name']}")
    print(f"🤖 Extracting...")
    
    result = extract_facility_facts_v2(
        hospital['name'],
        hospital['description'],
        client
    )
    
    total = (len(result.get('procedure',[])) + 
             len(result.get('equipment',[])) + 
             len(result.get('capability',[])))
    
    print(f"  📋 Procedures : {len(result.get('procedure',[]))}")
    print(f"  🔧 Equipment  : {len(result.get('equipment',[]))}")
    print(f"  ⚡ Capability : {len(result.get('capability',[]))}")
    print(f"  📊 Total facts: {total}")
    time.sleep(1)

# COMMAND ----------

import time
from datetime import datetime

print(f"🚀 Starting IDP extraction on {len(needs_extraction)} hospitals")
print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)

# Store results
extracted_results = []
success_count = 0
skip_count = 0
error_count = 0

for i, (idx, row) in enumerate(needs_extraction.iterrows()):
    
    # Progress update every 10 hospitals
    if i % 10 == 0:
        print(f"Progress: {i}/{len(needs_extraction)} hospitals processed...")
    
    # Skip if description is too short (less than 50 chars)
    if len(str(row['description'])) < 50:
        extracted_results.append({
            'unique_id': row['unique_id'],
            'name': row['name'],
            'extracted_procedure': '[]',
            'extracted_equipment': '[]',
            'extracted_capability': '[]',
            'extraction_status': 'SKIPPED_SHORT_DESC',
            'facts_count': 0
        })
        skip_count += 1
        continue
    
    # Run extraction
    result = extract_facility_facts_v2(
        row['name'],
        row['description'],
        client
    )
    
    # Check for errors
    if 'error' in result:
        status = 'ERROR'
        error_count += 1
    else:
        status = 'SUCCESS'
        success_count += 1
    
    # Calculate total facts
    total_facts = (len(result.get('procedure', [])) + 
                   len(result.get('equipment', [])) + 
                   len(result.get('capability', [])))
    
    extracted_results.append({
        'unique_id': row['unique_id'],
        'name': row['name'],
        'extracted_procedure': str(result.get('procedure', [])),
        'extracted_equipment': str(result.get('equipment', [])),
        'extracted_capability': str(result.get('capability', [])),
        'extraction_status': status,
        'facts_count': total_facts
    })
    
    # Small delay to avoid rate limiting
    time.sleep(0.5)

print(f"\n✅ EXTRACTION COMPLETE!")
print(f"   Success : {success_count}")
print(f"   Skipped : {skip_count}")
print(f"   Errors  : {error_count}")
print(f"   Total   : {len(extracted_results)}")
print(f"⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")

# COMMAND ----------

# Check what was saved
enriched_df = spark.table("enriched_facilities").toPandas()
print(f"Total rows: {len(enriched_df)}")
print(f"Has procedure: {(enriched_df['procedure'] != '[]').sum()}")
print(f"Has equipment: {(enriched_df['equipment'] != '[]').sum()}")
print(f"Has capability: {(enriched_df['capability'] != '[]').sum()}")
display(enriched_df[['name', 'procedure', 'equipment', 'capability']].head(5))

# COMMAND ----------

import pandas as pd

# Convert results to DataFrame
results_df = pd.DataFrame(extracted_results)

print(f"✅ Results in memory: {len(results_df)}")
print(f"   Success: {(results_df['extraction_status'] == 'SUCCESS').sum()}")
print(f"   Skipped: {(results_df['extraction_status'] == 'SKIPPED_SHORT_DESC').sum()}")
print(f"   Errors : {(results_df['extraction_status'] == 'ERROR').sum()}")

# Merge with original dataframe
df_merged = df.merge(
    results_df[['unique_id', 'extracted_procedure', 
                'extracted_equipment', 'extracted_capability',
                'extraction_status', 'facts_count']],
    on='unique_id',
    how='left'
)

# Fill extracted fields into original columns where original was empty
def fill_if_empty(original, extracted):
    if pd.isna(original) or str(original).strip() in ['', '[]', "['']"]:
        return extracted if extracted else original
    return original

df_merged['procedure'] = df_merged.apply(
    lambda r: fill_if_empty(r['procedure'], r['extracted_procedure']), axis=1)
df_merged['equipment'] = df_merged.apply(
    lambda r: fill_if_empty(r['equipment'], r['extracted_equipment']), axis=1)
df_merged['capability'] = df_merged.apply(
    lambda r: fill_if_empty(r['capability'], r['extracted_capability']), axis=1)

print(f"\n📊 BEFORE vs AFTER:")
print(f"   Procedure filled: {(df_merged['procedure'].notna() & (df_merged['procedure'] != '[]')).sum()}")
print(f"   Equipment filled: {(df_merged['equipment'].notna() & (df_merged['equipment'] != '[]')).sum()}")
print(f"   Capability filled: {(df_merged['capability'].notna() & (df_merged['capability'] != '[]')).sum()}")

# COMMAND ----------

# Save enriched data as Delta Table
spark_df = spark.createDataFrame(df_merged.astype(str).fillna(''))

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("enriched_facilities")

print("✅ Saved: enriched_facilities Delta table")
print(f"   Total rows: {spark_df.count()}")

# COMMAND ----------

# Check what was saved
enriched_df = spark.table("enriched_facilities").toPandas()
print(f"Total rows: {len(enriched_df)}")
print(f"Has procedure: {(enriched_df['procedure'] != '[]').sum()}")
print(f"Has equipment: {(enriched_df['equipment'] != '[]').sum()}")
print(f"Has capability: {(enriched_df['capability'] != '[]').sum()}")
display(enriched_df[['name', 'procedure', 'equipment', 'capability']].head(5))

# COMMAND ----------

import re

def extract_region_from_address(row):
    """
    Try to find region from existing address fields
    using simple text matching
    """
    
    # Ghana regions list
    ghana_regions = {
        # Greater Accra
        'accra': 'Greater Accra',
        'tema': 'Greater Accra',
        'east legon': 'Greater Accra',
        'dansoman': 'Greater Accra',
        'lapaz': 'Greater Accra',
        'madina': 'Greater Accra',
        'adabraka': 'Greater Accra',
        'osu': 'Greater Accra',
        'cantonments': 'Greater Accra',
        'airport': 'Greater Accra',
        'spintex': 'Greater Accra',
        'haatso': 'Greater Accra',
        'kasoa': 'Greater Accra',
        
        # Ashanti
        'kumasi': 'Ashanti',
        'ashanti': 'Ashanti',
        'obuasi': 'Ashanti',
        'bekwai': 'Ashanti',
        'konongo': 'Ashanti',
        'mampong': 'Ashanti',
        'ejisu': 'Ashanti',
        'asokwa': 'Ashanti',
        'afigya': 'Ashanti',
        
        # Western
        'takoradi': 'Western',
        'sekondi': 'Western',
        'western': 'Western',
        'tarkwa': 'Western',
        'axim': 'Western',
        'apremdo': 'Western',
        
        # Central
        'cape coast': 'Central',
        'central': 'Central',
        'winneba': 'Central',
        'kasoa': 'Central',
        'elmina': 'Central',
        
        # Eastern
        'koforidua': 'Eastern',
        'eastern': 'Eastern',
        'nkawkaw': 'Eastern',
        'suhum': 'Eastern',
        
        # Northern
        'tamale': 'Northern',
        'northern': 'Northern',
        'yendi': 'Northern',
        'savelugu': 'Northern',
        
        # Volta
        'ho': 'Volta',
        'volta': 'Volta',
        'hohoe': 'Volta',
        'keta': 'Volta',
        
        # Brong Ahafo
        'sunyani': 'Brong Ahafo',
        'brong': 'Brong Ahafo',
        'techiman': 'Brong Ahafo',
        'dormaa': 'Brong Ahafo',
        
        # Upper East
        'bolgatanga': 'Upper East',
        'upper east': 'Upper East',
        'navrongo': 'Upper East',
        'bawku': 'Upper East',
        
        # Upper West
        'wa': 'Upper West',
        'upper west': 'Upper West',
        'lawra': 'Upper West',
        
        # Savannah
        'damongo': 'Savannah',
        'savannah': 'Savannah',
        
        # North East
        'nalerigu': 'North East',
        'north east': 'North East',
        
        # Oti
        'dambai': 'Oti',
        'oti': 'Oti',
        
        # Ahafo
        'goaso': 'Ahafo',
        'ahafo': 'Ahafo',
        
        # Bono East
        'kintampo': 'Bono East',
        'bono east': 'Bono East',
        
        # Western North
        'sefwi': 'Western North',
        'western north': 'Western North',
        'bibiani': 'Western North',
    }
    
    # Combine all text fields to search through
    search_text = ' '.join([
        str(row.get('address_line1', '') or ''),
        str(row.get('address_line2', '') or ''),
        str(row.get('address_line3', '') or ''),
        str(row.get('address_city', '') or ''),
        str(row.get('description', '') or ''),
        str(row.get('source_url', '') or ''),
        str(row.get('name', '') or ''),
    ]).lower()
    
    # Search for region keywords
    for keyword, region in ghana_regions.items():
        if keyword in search_text:
            return region
    
    return 'Unknown'

print("✅ Region extractor function ready!")

# COMMAND ----------

# Load enriched data
enriched_df = spark.table("enriched_facilities").toPandas()

print(f"Before region fix:")
print(f"  Unknown regions: {(enriched_df['region_clean'] == 'Unknown').sum()}")
print(f"  Known regions  : {(enriched_df['region_clean'] != 'Unknown').sum()}")

# Apply region extraction to Unknown rows
unknown_mask = enriched_df['region_clean'] == 'Unknown'

enriched_df.loc[unknown_mask, 'region_clean'] = enriched_df[unknown_mask].apply(
    extract_region_from_address, axis=1
)

print(f"\nAfter region fix:")
print(f"  Unknown regions: {(enriched_df['region_clean'] == 'Unknown').sum()}")
print(f"  Known regions  : {(enriched_df['region_clean'] != 'Unknown').sum()}")

print(f"\n=== REGION COUNTS AFTER FIX ===")
region_counts = enriched_df['region_clean'].value_counts()
for region, count in region_counts.items():
    bar = "█" * int(count/3)
    print(f"  {region:<20} {bar} ({count})")

# COMMAND ----------

def extract_region_with_ai(row, client):
    """Use Groq AI to extract region from description"""
    
    # Only process if we have some text to work with
    text = str(row.get('description', '') or '')
    address = str(row.get('address_line1', '') or '')
    city = str(row.get('address_city', '') or '')
    name = str(row.get('name', '') or '')
    
    if len(text + address + city) < 20:
        return 'Unknown'
    
    prompt = f"""
Given this information about a Ghana healthcare facility:

Name: {name}
City: {city}
Address: {address}
Description: {text[:300]}

Which Ghana region is this facility in?
Choose ONLY from these options:
Greater Accra, Ashanti, Western, Central, Eastern, 
Northern, Volta, Brong Ahafo, Upper East, Upper West,
Savannah, North East, Oti, Ahafo, Bono East, Western North

Reply with ONLY the region name, nothing else.
If you cannot determine the region, reply: Unknown
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        region = response.choices[0].message.content.strip()
        
        # Validate it's a real region
        valid_regions = [
            'Greater Accra', 'Ashanti', 'Western', 'Central', 
            'Eastern', 'Northern', 'Volta', 'Brong Ahafo', 
            'Upper East', 'Upper West', 'Savannah', 'North East',
            'Oti', 'Ahafo', 'Bono East', 'Western North'
        ]
        
        if region in valid_regions:
            return region
        return 'Unknown'
        
    except:
        return 'Unknown'

# Apply AI extraction to remaining Unknown rows
still_unknown = enriched_df['region_clean'] == 'Unknown'
unknown_count = still_unknown.sum()

print(f"Using AI to fix {unknown_count} remaining unknown regions...")
print("This may take a few minutes...")

for i, idx in enumerate(enriched_df[still_unknown].index):
    if i % 20 == 0:
        print(f"  Progress: {i}/{unknown_count}")
    
    enriched_df.loc[idx, 'region_clean'] = extract_region_with_ai(
        enriched_df.loc[idx], client
    )
    time.sleep(0.3)

print(f"\n✅ AI region extraction complete!")
print(f"\nFinal region counts:")
print(f"  Unknown : {(enriched_df['region_clean'] == 'Unknown').sum()}")
print(f"  Known   : {(enriched_df['region_clean'] != 'Unknown').sum()}")

# COMMAND ----------

# Save final enriched data with fixed regions
spark_df = spark.createDataFrame(enriched_df.astype(str).fillna(''))

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("enriched_facilities")

print("✅ Final enriched data saved!")
print(f"   Total rows     : {spark_df.count()}")
print(f"   Known regions  : {(enriched_df['region_clean'] != 'Unknown').sum()}")
print(f"   Unknown regions: {(enriched_df['region_clean'] == 'Unknown').sum()}")
print("\n🎉 Data is now ready for Gap Analysis!")