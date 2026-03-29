# Databricks notebook source
import pandas as pd
import ast

# Load enriched data
df = spark.table("enriched_facilities").toPandas()

print(f"✅ Loaded: {len(df)} facilities")
print(f"   Known regions : {(df['region_clean'] != 'Unknown').sum()}")
print(f"   Unknown regions: {(df['region_clean'] == 'Unknown').sum()}")
print(f"\nRegion counts:")
print(df['region_clean'].value_counts().head(10).to_string())

# COMMAND ----------

# Check what keywords exist in our extracted data
import ast

def parse_list(val):
    try:
        if pd.isna(val) or val == '':
            return []
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []

# Parse all capability/procedure/equipment into proper lists
df['cap_list']  = df['capability'].apply(parse_list)
df['proc_list'] = df['procedure'].apply(parse_list)
df['equip_list'] = df['equipment'].apply(parse_list)

# Combine all text for keyword search
df['all_text'] = df.apply(
    lambda r: ' '.join(r['cap_list'] + r['proc_list'] + r['equip_list']).lower(), 
    axis=1
)

# Check critical medical keywords
keywords = {
    'ICU': ['icu', 'intensive care', 'critical care'],
    'Emergency': ['emergency', 'accident', '24/7', '24 hour'],
    'Surgery': ['surgery', 'surgical', 'operating theatre', 'theatre'],
    'Maternity': ['maternity', 'obstetric', 'gynecology', 'delivery'],
    'Laboratory': ['laboratory', 'lab test', 'diagnostic lab'],
    'Imaging': ['x-ray', 'xray', 'ultrasound', 'scan', 'mri', 'ct scan'],
    'Pediatrics': ['pediatric', 'children', 'child care'],
    'Pharmacy': ['pharmacy', 'pharmaceutical'],
}

print("=== KEYWORD COVERAGE ACROSS ALL HOSPITALS ===\n")
for service, keys in keywords.items():
    mask = df['all_text'].apply(
        lambda t: any(k in t for k in keys)
    )
    count = mask.sum()
    pct = round(count/len(df)*100, 1)
    bar = "█" * int(pct/5)
    print(f"  {service:<12} {bar} {count} hospitals ({pct}%)")

# COMMAND ----------

# Score each region on critical services
def has_service(text, keywords):
    return any(k in text.lower() for k in keywords)

# Define critical services
services = {
    'has_icu':       ['icu', 'intensive care', 'critical care'],
    'has_emergency': ['emergency', 'accident', '24/7', '24 hour'],
    'has_surgery':   ['surgery', 'surgical', 'operating theatre'],
    'has_maternity': ['maternity', 'obstetric', 'gynecology', 'delivery'],
    'has_lab':       ['laboratory', 'lab test', 'diagnostic'],
    'has_imaging':   ['x-ray', 'xray', 'ultrasound', 'scan', 'mri'],
    'has_pediatrics':['pediatric', 'children', 'child care'],
    'has_pharmacy':  ['pharmacy', 'pharmaceutical'],
}

# Add service flags to each hospital
for service, keys in services.items():
    df[service] = df['all_text'].apply(lambda t: has_service(t, keys))

# Group by region
region_scores = df.groupby('region_clean').agg(
    total_facilities = ('name', 'count'),
    has_icu          = ('has_icu', 'sum'),
    has_emergency    = ('has_emergency', 'sum'),
    has_surgery      = ('has_surgery', 'sum'),
    has_maternity    = ('has_maternity', 'sum'),
    has_lab          = ('has_lab', 'sum'),
    has_imaging      = ('has_imaging', 'sum'),
    has_pediatrics   = ('has_pediatrics', 'sum'),
    has_pharmacy     = ('has_pharmacy', 'sum'),
).reset_index()

# Calculate desert score (0-8, lower = worse)
region_scores['services_available'] = (
    (region_scores['has_icu'] > 0).astype(int) +
    (region_scores['has_emergency'] > 0).astype(int) +
    (region_scores['has_surgery'] > 0).astype(int) +
    (region_scores['has_maternity'] > 0).astype(int) +
    (region_scores['has_lab'] > 0).astype(int) +
    (region_scores['has_imaging'] > 0).astype(int) +
    (region_scores['has_pediatrics'] > 0).astype(int) +
    (region_scores['has_pharmacy'] > 0).astype(int)
)

# Risk level
def risk_level(row):
    score = row['services_available']
    facilities = row['total_facilities']
    
    if score <= 2 or facilities <= 3:
        return '🔴 CRITICAL DESERT'
    elif score <= 4 or facilities <= 8:
        return '🟠 HIGH RISK'
    elif score <= 6 or facilities <= 20:
        return '🟡 MODERATE RISK'
    else:
        return '🟢 ADEQUATE'

region_scores['risk_level'] = region_scores.apply(risk_level, axis=1)
region_scores = region_scores.sort_values('services_available')

print("=== MEDICAL DESERT ANALYSIS ===\n")
print(f"{'Region':<20} {'Facilities':>10} {'Services':>9} {'ICU':>4} {'ER':>4} {'Surg':>5} {'Risk'}")
print("─" * 80)

for _, row in region_scores.iterrows():
    if row['region_clean'] == 'Unknown':
        continue
    print(
        f"{row['region_clean']:<20} "
        f"{row['total_facilities']:>10} "
        f"{row['services_available']:>9}/8 "
        f"{'✅' if row['has_icu'] > 0 else '❌':>4} "
        f"{'✅' if row['has_emergency'] > 0 else '❌':>4} "
        f"{'✅' if row['has_surgery'] > 0 else '❌':>5} "
        f"  {row['risk_level']}"
    )

# COMMAND ----------

# Generate NGO action recommendations for each desert region
print("=== NGO RESOURCE ROUTING RECOMMENDATIONS ===\n")

critical = region_scores[
    region_scores['risk_level'] == '🔴 CRITICAL DESERT'
].sort_values('services_available')

for _, row in critical.iterrows():
    if row['region_clean'] == 'Unknown':
        continue
    
    # What's missing
    missing = []
    if row['has_icu'] == 0:      missing.append('ICU Unit')
    if row['has_emergency'] == 0: missing.append('Emergency Care')
    if row['has_surgery'] == 0:   missing.append('Surgery')
    if row['has_maternity'] == 0: missing.append('Maternity')
    if row['has_lab'] == 0:       missing.append('Laboratory')
    if row['has_imaging'] == 0:   missing.append('Imaging/X-Ray')
    if row['has_pediatrics'] == 0:missing.append('Pediatrics')
    if row['has_pharmacy'] == 0:  missing.append('Pharmacy')
    
    print(f"📍 {row['region_clean']}")
    print(f"   Facilities : {int(row['total_facilities'])}")
    print(f"   Services   : {int(row['services_available'])}/8")
    print(f"   Missing    : {', '.join(missing[:4])}")
    print(f"   Action     : Deploy medical team with {missing[0] if missing else 'support'}")
    print()

# COMMAND ----------

# Save results
spark_df = spark.createDataFrame(region_scores.astype(str))

spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("region_gap_analysis")

print("✅ Gap analysis saved: region_gap_analysis")
print(f"\n📊 SUMMARY:")
print(f"  🔴 Critical Deserts : {(region_scores['risk_level'] == '🔴 CRITICAL DESERT').sum()} regions")
print(f"  🟠 High Risk        : {(region_scores['risk_level'] == '🟠 HIGH RISK').sum()} regions")
print(f"  🟡 Moderate Risk    : {(region_scores['risk_level'] == '🟡 MODERATE RISK').sum()} regions")
print(f"  🟢 Adequate         : {(region_scores['risk_level'] == '🟢 ADEQUATE').sum()} regions")
print(f"\n🎉 Notebook 3 COMPLETE!")

# COMMAND ----------

# Flag suspicious/anomalous hospital claims
print("=== ANOMALY DETECTION ===\n")

anomalies = []

for _, row in df.iterrows():
    hospital_anomalies = []
    
    # Get facility type
    ftype = str(row.get('facilityTypeId', '')).lower()
    all_text = str(row.get('all_text', '')).lower()
    cap_count = len(row.get('cap_list', []))
    
    # Anomaly 1: Small clinic claiming ICU
    if ftype == 'clinic' and 'icu' in all_text:
        hospital_anomalies.append('⚠️ Clinic claiming ICU — verify')
    
    # Anomaly 2: Pharmacy claiming surgery
    if ftype == 'pharmacy' and 'surgery' in all_text:
        hospital_anomalies.append('⚠️ Pharmacy claiming surgery — suspicious')
    
    # Anomaly 3: Hospital with zero capabilities
    if ftype == 'hospital' and cap_count == 0:
        hospital_anomalies.append('⚠️ Hospital with no capabilities listed')
    
    # Anomaly 4: Claiming ICU but no emergency
    if 'icu' in all_text and 'emergency' not in all_text:
        hospital_anomalies.append('⚠️ Has ICU but no emergency — inconsistent')
    
    # Anomaly 5: No location data
    if row.get('region_clean') == 'Unknown':
        hospital_anomalies.append('⚠️ No location data — cannot map')
    
    if hospital_anomalies:
        anomalies.append({
            'name': row['name'],
            'city': row['address_city'],
            'region': row['region_clean'],
            'facility_type': ftype,
            'anomalies': ' | '.join(hospital_anomalies)
        })

anomaly_df = pd.DataFrame(anomalies)
print(f"Total anomalies found: {len(anomaly_df)}")
print(f"\nTop anomalies:")
print(anomaly_df['anomalies'].value_counts().head(10).to_string())

# Save anomalies
spark.createDataFrame(anomaly_df.astype(str)) \
     .write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("facility_anomalies")

print(f"\n✅ Anomalies saved to: facility_anomalies")

# COMMAND ----------

# Install folium for map
%pip install folium -q

# COMMAND ----------

import folium

# Ghana region coordinates
region_coords = {
    'Greater Accra': (5.6037, -0.1870),
    'Ashanti':       (6.7470, -1.5209),
    'Western':       (5.1176, -2.0452),
    'Central':       (5.5071, -1.0033),
    'Eastern':       (6.5403, -0.4678),
    'Northern':      (9.5404, -0.9062),
    'Volta':         (7.9465, -0.5265),
    'Brong Ahafo':   (7.9408, -1.7680),
    'Upper East':    (10.7877, -0.8512),
    'Upper West':    (10.2529, -2.3267),
    'Savannah':      (9.0820, -1.6596),
    'North East':    (10.5500, -0.3667),
    'Oti':           (8.0000,  0.1500),
    'Ahafo':         (7.3333, -2.3333),
    'Bono East':     (7.7500, -1.0500),
    'Western North': (6.3000, -2.8000),
}

color_map = {
    '🔴 CRITICAL DESERT': 'red',
    '🟠 HIGH RISK':        'orange',
    '🟡 MODERATE RISK':    'yellow',
    '🟢 ADEQUATE':         'green',
}

# Create map
ghana_map = folium.Map(
    location=[7.9465, -1.0232],
    zoom_start=7,
    tiles='CartoDB positron'
)

# Title
title_html = '''
<h3 align="center" style="font-size:20px;color:#333;margin-top:10px">
    🏥 Ghana Medical Desert Map — Virtue Foundation IDP Agent
</h3>
'''
ghana_map.get_root().html.add_child(folium.Element(title_html))

# Load region data
region_data = spark.table("region_gap_analysis").toPandas()

plotted = 0
for _, row in region_data.iterrows():
    region = row['region_clean']
    if region not in region_coords:
        continue
    
    lat, lon = region_coords[region]
    risk = row['risk_level']
    color = color_map.get(risk, 'gray')
    
    missing = []
    if str(row['has_icu']) == '0':        missing.append('ICU')
    if str(row['has_emergency']) == '0':  missing.append('Emergency')
    if str(row['has_surgery']) == '0':    missing.append('Surgery')
    if str(row['has_maternity']) == '0':  missing.append('Maternity')
    if str(row['has_lab']) == '0':        missing.append('Lab')
    if str(row['has_imaging']) == '0':    missing.append('Imaging')
    if str(row['has_pediatrics']) == '0': missing.append('Pediatrics')
    if str(row['has_pharmacy']) == '0':   missing.append('Pharmacy')

    popup_html = f"""
    <div style="width:220px;font-family:Arial">
        <h4 style="color:#333">{region}</h4>
        <hr>
        <b>Risk:</b> {risk}<br>
        <b>Facilities:</b> {row['total_facilities']}<br>
        <b>Services:</b> {row['services_available']}/8<br>
        <hr>
        <b>ICU:</b> {'✅' if str(row['has_icu'])!='0' else '❌'}&nbsp;
        <b>ER:</b> {'✅' if str(row['has_emergency'])!='0' else '❌'}&nbsp;
        <b>Surgery:</b> {'✅' if str(row['has_surgery'])!='0' else '❌'}<br>
        <b>Lab:</b> {'✅' if str(row['has_lab'])!='0' else '❌'}&nbsp;
        <b>Imaging:</b> {'✅' if str(row['has_imaging'])!='0' else '❌'}<br>
        <hr>
        <b style="color:red">Missing: {', '.join(missing) if missing else 'None ✅'}</b>
    </div>
    """
    
    radius = max(8, min(40, int(row['total_facilities'])//5))
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"📍 {region} | {risk} | {row['total_facilities']} facilities"
    ).add_to(ghana_map)
    plotted += 1

# Legend
legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
     background:white;padding:15px;border-radius:10px;
     border:2px solid #ccc;font-size:13px;font-family:Arial">
    <b>🏥 Medical Desert Risk</b><br><br>
    <span style="color:red;font-size:18px">●</span> Critical Desert<br>
    <span style="color:orange;font-size:18px">●</span> High Risk<br>
    <span style="color:#cccc00;font-size:18px">●</span> Moderate Risk<br>
    <span style="color:green;font-size:18px">●</span> Adequate Coverage<br>
    <br><small>Circle size = number of facilities</small>
</div>
'''
ghana_map.get_root().html.add_child(folium.Element(legend_html))

ghana_map.save('/tmp/ghana_map.html')
print(f"✅ Map created! {plotted} regions plotted!")
ghana_map

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/cp4707@srmist.edu.in/medical-desert-idp-agent")

with mlflow.start_run(run_name="Ghana_IDP_Extraction"):
    
    # Metrics
    mlflow.log_metric("total_facilities", 987)
    mlflow.log_metric("facilities_extracted", 440)
    mlflow.log_metric("procedure_filled", 848)
    mlflow.log_metric("equipment_filled", 765)
    mlflow.log_metric("capability_filled", 976)
    mlflow.log_metric("regions_mapped", 955)
    mlflow.log_metric("critical_deserts", 26)
    mlflow.log_metric("anomalies_detected", 103)
    
    # Parameters
    mlflow.log_param("ai_model", "llama-3.3-70b-versatile")
    mlflow.log_param("ai_provider", "Groq")
    mlflow.log_param("prompt_version", "v2")
    mlflow.log_param("country", "Ghana")
    mlflow.log_param("total_anomaly_types", 5)
    
    # Save map as artifact
    mlflow.log_artifact("/tmp/ghana_map.html")
    
    print("✅ MLflow experiment tracked!")
    print("\n📊 WHAT WAS LOGGED:")
    print("   Metrics  → extraction rates, desert counts")
    print("   Params   → model used, prompt version")
    print("   Artifact → Ghana medical desert map")
    print("\n👉 View in: Experiments → medical-desert-idp-agent")