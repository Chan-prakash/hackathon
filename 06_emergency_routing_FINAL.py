# Databricks notebook source
# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — Emergency Patient Routing + Doctor Deployment
# Ghana Medical Desert Agent | Virtue Foundation
# Unique features: ambulance routing, doctor deployment optimizer
# ══════════════════════════════════════════════════════════════════

# COMMAND ----------

# MAGIC %pip install groq faiss-cpu sentence-transformers -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, json, time, math
import pandas as pd
import numpy as np
import faiss
import mlflow
from groq import Groq
from sentence_transformers import SentenceTransformer

GROQ_KEY = os.environ.get("GROQ_KEY")
client   = Groq(api_key=GROQ_KEY)

# Load all data
hospital_metadata = spark.table("hospital_metadata_full").toPandas().fillna("")
region_gap        = spark.table("region_gap_analysis").toPandas().fillna("")

print(f"✅ Loaded {len(hospital_metadata)} hospitals")
print(f"✅ Loaded {len(region_gap)} regions for gap analysis")

# COMMAND ----------

# ── Ghana region GPS coordinates ────────────────────────────────
REGION_COORDS = {
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

# ── Medical condition → required capability mapping ─────────────
CONDITION_REQUIREMENTS = {
    "cardiac arrest":         ["icu", "emergency", "cardiology", "cardiac"],
    "stroke":                 ["emergency", "neurology", "icu", "imaging"],
    "severe trauma":          ["emergency", "surgery", "icu", "operating"],
    "complicated pregnancy":  ["maternity", "obstetric", "gynecology", "emergency"],
    "pediatric emergency":    ["pediatric", "emergency", "children"],
    "eye injury":             ["ophthalmology", "eye", "ophthalm"],
    "fracture":               ["surgery", "orthopedic", "imaging", "x-ray"],
    "burns":                  ["emergency", "surgery", "icu", "burn"],
    "kidney failure":         ["nephrology", "dialysis", "icu"],
    "mental health crisis":   ["psychiatry", "mental", "psychiatric"],
    "general emergency":      ["emergency", "24/7", "accident"],
    "surgery needed":         ["surgery", "surgical", "operating theatre"],
}

print("✅ Region coordinates and condition mappings loaded")

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════
# FEATURE 1: EMERGENCY PATIENT ROUTING
# Given: patient location (region) + medical condition
# Returns: nearest capable hospital + estimated travel time
# ══════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two GPS coordinates."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def hospital_has_capability(row, required_keywords):
    """Check if hospital text contains any required keyword."""
    text = ' '.join([
        str(row.get('capability',    '')),
        str(row.get('procedure',     '')),
        str(row.get('equipment',     '')),
        str(row.get('specialties',   '')),
        str(row.get('description',   '')),
    ]).lower()
    return any(kw in text for kw in required_keywords)

def find_nearest_capable_hospital(patient_region, condition, max_results=5):
    """
    Core emergency routing function.
    Finds nearest hospitals that can treat the given condition,
    sorted by distance from patient region.
    """
    condition_lower = condition.lower()

    # Get required keywords for this condition
    required_keywords = []
    for cond_name, keywords in CONDITION_REQUIREMENTS.items():
        if cond_name in condition_lower or any(k in condition_lower for k in keywords):
            required_keywords.extend(keywords)

    if not required_keywords:
        required_keywords = ["emergency", "hospital"]  # fallback

    # Get patient region coordinates
    patient_coords = REGION_COORDS.get(patient_region)
    if not patient_coords:
        return None, f"Region '{patient_region}' coordinates not found"

    pat_lat, pat_lon = patient_coords

    # Score each hospital
    candidates = []
    for _, row in hospital_metadata.iterrows():
        hosp_region = row.get('region_clean', 'Unknown')
        if hosp_region == 'Unknown':
            continue

        region_coords = REGION_COORDS.get(hosp_region)
        if not region_coords:
            continue

        # Check capability match
        if not hospital_has_capability(row, required_keywords):
            continue

        dist_km = haversine_km(pat_lat, pat_lon, region_coords[0], region_coords[1])

        # Confidence scoring
        confidence = row.get('confidence', 'Medium')
        conf_score = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4}.get(confidence, 0.5)

        # Travel time estimate (Ghana avg road speed ~50 km/h)
        travel_hours = dist_km / 50.0

        candidates.append({
            'name':          row['name'],
            'region':        hosp_region,
            'city':          row.get('address_city', ''),
            'type':          row.get('facilityTypeId', ''),
            'confidence':    confidence,
            'conf_score':    conf_score,
            'distance_km':   round(dist_km, 1),
            'travel_hours':  round(travel_hours, 1),
            'travel_mins':   round(travel_hours * 60),
            'capability':    str(row.get('capability', ''))[:200],
            'procedure':     str(row.get('procedure',  ''))[:200],
        })

    # Sort by distance
    candidates.sort(key=lambda x: x['distance_km'])
    return candidates[:max_results], None

print("✅ Emergency routing function ready")

# COMMAND ----------

# ── Generate AI routing recommendation ─────────────────────────
def get_emergency_recommendation(patient_region, condition, hospitals):
    """Use Groq to generate a human-readable emergency routing recommendation."""
    if not hospitals:
        return "No capable hospitals found for this condition in Ghana."

    hosp_context = ""
    for i, h in enumerate(hospitals[:3], 1):
        hosp_context += f"""
Hospital {i}: {h['name']}
  Region: {h['region']} | City: {h['city']}
  Distance: {h['distance_km']} km (~{h['travel_mins']} min drive)
  Confidence: {h['confidence']}
  Capabilities: {h['capability'][:150]}
"""

    prompt = f"""You are an emergency medical coordinator for Ghana NGO healthcare.

EMERGENCY SITUATION:
Patient location: {patient_region}
Medical condition: {condition}

NEAREST CAPABLE HOSPITALS:
{hosp_context}

Provide a clear emergency routing recommendation:
1. Which hospital to go to FIRST and why
2. Estimated travel time and urgency level (CRITICAL/HIGH/MODERATE)
3. What to prepare/bring
4. Any warning about data confidence

Be concise and actionable. Under 150 words."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, n=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI recommendation unavailable: {str(e)[:50]}"

print("✅ AI recommendation function ready")

# COMMAND ----------

# ── Test emergency routing ──────────────────────────────────────
print("=" * 65)
print("EMERGENCY ROUTING TEST SCENARIOS")
print("=" * 65)

test_cases = [
    ("Upper East",    "cardiac arrest"),
    ("Northern",      "complicated pregnancy"),
    ("Upper West",    "severe trauma"),
    ("Volta",         "stroke"),
    ("Greater Accra", "pediatric emergency"),
]

routing_results = []

for patient_region, condition in test_cases:
    print(f"\n🚨 EMERGENCY: {condition.upper()}")
    print(f"   Patient location: {patient_region}")

    hospitals, error = find_nearest_capable_hospital(patient_region, condition)

    if error:
        print(f"   ❌ Error: {error}")
        continue

    if not hospitals:
        print(f"   ⚠️  No capable hospitals found — CRITICAL DESERT confirmed!")
        continue

    nearest = hospitals[0]
    print(f"\n   🏥 NEAREST CAPABLE HOSPITAL:")
    print(f"      Name       : {nearest['name']}")
    print(f"      Region     : {nearest['region']}")
    print(f"      Distance   : {nearest['distance_km']} km")
    print(f"      Travel time: ~{nearest['travel_mins']} minutes")
    print(f"      Confidence : {nearest['confidence']}")

    if len(hospitals) > 1:
        print(f"\n   📋 OTHER OPTIONS ({len(hospitals)-1} more):")
        for h in hospitals[1:3]:
            print(f"      • {h['name']} [{h['region']}] — {h['distance_km']} km")

    recommendation = get_emergency_recommendation(patient_region, condition, hospitals)
    print(f"\n   💬 AI RECOMMENDATION:")
    print(f"      {recommendation}")

    routing_results.append({
        "patient_region": patient_region,
        "condition":      condition,
        "nearest_hospital": nearest['name'],
        "distance_km":    nearest['distance_km'],
        "travel_mins":    nearest['travel_mins'],
        "recommendation": recommendation,
    })

    print("─" * 65)
    time.sleep(1)

print(f"\n✅ Emergency routing tested on {len(test_cases)} scenarios!")

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════
# FEATURE 2: DOCTOR DEPLOYMENT OPTIMIZER
# Identifies deserts + finds nearest region with surplus
# Recommends specific deployments
# ══════════════════════════════════════════════════════════════════

def get_deployment_recommendations():
    """
    Finds critical desert regions and recommends doctor deployments
    from nearby regions with surplus capacity.
    """
    # Identify critical desert regions (no surgery or no emergency)
    deserts = region_gap[
        (region_gap['risk_level'].str.contains('CRITICAL', na=False)) &
        (region_gap['region_clean'] != 'Unknown')
    ].copy()

    # Identify well-covered regions (potential surplus)
    surplus_regions = region_gap[
        region_gap['risk_level'] == '🟢 ADEQUATE'
    ].copy()

    deployments = []

    for _, desert_row in deserts.iterrows():
        desert_region = desert_row['region_clean']
        desert_coords = REGION_COORDS.get(desert_region)
        if not desert_coords:
            continue

        # What's missing
        missing_services = []
        if str(desert_row.get('has_icu', '0')) == '0':       missing_services.append('ICU specialist')
        if str(desert_row.get('has_surgery', '0')) == '0':   missing_services.append('Surgeon')
        if str(desert_row.get('has_emergency', '0')) == '0': missing_services.append('Emergency doctor')
        if str(desert_row.get('has_maternity', '0')) == '0': missing_services.append('Obstetrician')
        if str(desert_row.get('has_pediatrics','0')) == '0': missing_services.append('Pediatrician')

        # Find nearest surplus region
        nearest_surplus = None
        min_dist = float('inf')

        for _, surplus_row in surplus_regions.iterrows():
            surplus_region = surplus_row['region_clean']
            surplus_coords = REGION_COORDS.get(surplus_region)
            if not surplus_coords:
                continue
            dist = haversine_km(desert_coords[0], desert_coords[1],
                                surplus_coords[0], surplus_coords[1])
            if dist < min_dist:
                min_dist = dist
                nearest_surplus = surplus_region

        deployments.append({
            'desert_region':    desert_region,
            'facilities':       int(desert_row['total_facilities']),
            'services':         int(desert_row['services_available']),
            'missing_services': missing_services,
            'source_region':    nearest_surplus or 'Greater Accra',
            'distance_km':      round(min_dist, 1),
            'priority':         '🔴 URGENT' if int(desert_row['services_available']) <= 1 else '🟠 HIGH',
        })

    return sorted(deployments, key=lambda x: x['services'])

deployment_plan = get_deployment_recommendations()

print("=" * 65)
print("DOCTOR DEPLOYMENT OPTIMIZER")
print("=" * 65)

for dep in deployment_plan:
    print(f"\n📍 {dep['desert_region']} ({dep['priority']})")
    print(f"   Facilities : {dep['facilities']}")
    print(f"   Services   : {dep['services']}/8")
    print(f"   Missing    : {', '.join(dep['missing_services'][:3])}")
    print(f"   Deploy from: {dep['source_region']} ({dep['distance_km']} km away)")
    print(f"   Action     : Send {dep['missing_services'][0] if dep['missing_services'] else 'support team'}")
    print(f"               to {dep['desert_region']} for 2-week rotation")

print(f"\n✅ Deployment plan: {len(deployment_plan)} regions need immediate support")

# COMMAND ----------

# ── Generate AI deployment summary ─────────────────────────────
def get_ai_deployment_summary(deployment_plan):
    """AI-generated strategic summary for NGO leadership."""
    plan_text = ""
    for dep in deployment_plan[:5]:
        plan_text += f"\n- {dep['desert_region']}: needs {', '.join(dep['missing_services'][:2])}, nearest source {dep['source_region']} ({dep['distance_km']}km)"

    prompt = f"""You are advising the Virtue Foundation NGO leadership on healthcare deployment in Ghana.

CURRENT MEDICAL DESERT SITUATION:
{plan_text}

Write a strategic deployment memo (under 200 words) that:
1. Summarizes the most urgent deployments
2. Recommends the top 3 immediate actions
3. Highlights the overall impact if these deployments happen
4. Uses specific region names and distances

Keep it actionable and suitable for NGO leadership."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, n=1,
    )
    return response.choices[0].message.content.strip()

ai_summary = get_ai_deployment_summary(deployment_plan)

print("\n" + "=" * 65)
print("AI STRATEGIC DEPLOYMENT MEMO — Virtue Foundation")
print("=" * 65)
print(ai_summary)

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════
# FEATURE 3: BEFORE vs AFTER IDP IMPACT COMPARISON
# Shows judges exactly what the AI extraction accomplished
# ══════════════════════════════════════════════════════════════════

original_df  = spark.table("medical_facilities_clean").toPandas().fillna("")
enriched_df  = spark.table("enriched_facilities").toPandas().fillna("")

def completeness(df):
    has_proc  = (df['procedure'].str.len() > 5).sum()
    has_equip = (df['equipment'].str.len() > 5).sum()
    has_cap   = (df['capability'].str.len() > 5).sum()
    complete  = ((df['procedure'].str.len() > 5) &
                 (df['equipment'].str.len() > 5) &
                 (df['capability'].str.len() > 5)).sum()
    return has_proc, has_equip, has_cap, complete

b_proc, b_equip, b_cap, b_complete = completeness(original_df)
a_proc, a_equip, a_cap, a_complete = completeness(enriched_df)

total = len(original_df)

print("=" * 65)
print("BEFORE vs AFTER IDP AGENT — IMPACT COMPARISON")
print("=" * 65)
print(f"\n{'Metric':<30} {'BEFORE':>10} {'AFTER':>10} {'Gain':>10}")
print("─" * 65)
print(f"{'Has Procedure':<30} {b_proc:>9} {a_proc:>10} {a_proc-b_proc:>+10}")
print(f"{'Has Equipment':<30} {b_equip:>9} {a_equip:>10} {a_equip-b_equip:>+10}")
print(f"{'Has Capability':<30} {b_cap:>9} {a_cap:>10} {a_cap-b_cap:>+10}")
print(f"{'Fully Complete Hospitals':<30} {b_complete:>9} {a_complete:>10} {a_complete-b_complete:>+10}")
print(f"\n  BEFORE: {round(b_complete/total*100,1)}% hospitals fully complete")
print(f"  AFTER : {round(a_complete/total*100,1)}% hospitals fully complete")
print(f"  ✅ IDP agent UNLOCKED {a_complete - b_complete} additional hospitals for analysis!")

# COMMAND ----------

import os
import mlflow
import pandas as pd

# 1. SAVE DATA TO DELTA (This part is working correctly)
# ------------------------------------------------------------------
try:
    routing_df = pd.DataFrame(routing_results)
    spark.createDataFrame(routing_df.astype(str)) \
         .write.format("delta").mode("overwrite") \
         .option("overwriteSchema", "true") \
         .saveAsTable("emergency_routing_results")

    deployment_df = pd.DataFrame([
        {**d, 'missing_services': ', '.join(d['missing_services'])} 
        for d in deployment_plan
    ])
    spark.createDataFrame(deployment_df.astype(str)) \
         .write.format("delta").mode("overwrite") \
         .option("overwriteSchema", "true") \
         .saveAsTable("doctor_deployment_plan")

    print("✅ Saved: emergency_routing_results")
    print("✅ Saved: doctor_deployment_plan")
except Exception as e:
    print(f"❌ Data Save Error: {e}")

# 2. MLFLOW TRACKING (SILENT MODE FOR SERVERLESS)
# ------------------------------------------------------------------
# We use a broad try-except to prevent the gRPC error from stopping the notebook
try:
    # Do not set any URIs. Serverless handles this internally or not at all.
    with mlflow.start_run(run_name="Emergency_Routing_v1"):
        mlflow.log_param("project", "Ghana Medical Desert")
        mlflow.log_metric("unlocked_hospitals", a_complete - b_complete)
        
        # Log local CSVs as backup
        deployment_df.to_csv("/tmp/deploy.csv", index=False)
        mlflow.log_artifact("/tmp/deploy.csv")
        print("✅ MLflow tracked successfully.")
except:
    # If Serverless blocks MLflow entirely, we ignore it so the notebook finishes
    print("⚠️ MLflow Tracking skipped (Compute restriction). Data was still saved to Delta.")

# 3. FINAL PROJECT SUMMARY
# ------------------------------------------------------------------
print(f"\n🎉 Notebook 6 COMPLETE!")
print(f"   Feature 1 — Emergency Patient Routing : ✅")
print(f"   Feature 2 — Doctor Deployment Optimizer: ✅")
print(f"   Feature 3 — Before vs After IDP Impact : ✅")

# COMMAND ----------

# Add this at the end of Notebook 6 after everything runs
import pandas as pd

routing_df = spark.table("emergency_routing_results").toPandas()
deployment_df = spark.table("doctor_deployment_plan").toPandas()

routing_df.to_csv("/tmp/emergency_routing_results.csv", index=False)
deployment_df.to_csv("/tmp/doctor_deployment_plan.csv", index=False)

print("✅ Download these two files and push to GitHub:")
print("   /tmp/emergency_routing_results.csv")
print("   /tmp/doctor_deployment_plan.csv")

# COMMAND ----------

# 1. Define the correct path based on your Catalog sidebar
volume_path = "/Volumes/workspace/default/project"

# 2. Save the files directly to the volume
# Note: We use the 'file:' prefix for local-style API calls to Volumes
routing_df.to_csv(f"{volume_path}/emergency_routing_results.csv", index=False)
deployment_df.to_csv(f"{volume_path}/doctor_deployment_plan.csv", index=False)

print(f"✅ Success! Files saved to: {volume_path}")