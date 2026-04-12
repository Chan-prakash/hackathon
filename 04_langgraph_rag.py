# Databricks notebook source
# Cell 1 — Install with correct versions
# MAGIC %pip install --upgrade typing_extensions groq langgraph langchain langchain-groq faiss-cpu sentence-transformers -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import time
import pandas as pd
from typing import TypedDict, List
from groq import Groq
from langgraph.graph import StateGraph, END

GROQ_KEY = os.environ.get("GROQ_KEY")  # Set in Databricks cluster env vars
client = Groq(api_key=GROQ_KEY)

print("✅ Imports done!")
print("✅ Groq connected!")
print("✅ LangGraph ready!")

# COMMAND ----------

# Cell 3 — Define the State (data that flows through the pipeline)

class HospitalState(TypedDict):
    """
    This is like a shared clipboard that every node reads and writes to.
    Node 1 fills in extracted facts.
    Node 2 fills in anomalies.
    Node 3 fills in the NGO answer.
    """
    # INPUT
    hospital_name: str
    description: str
    facility_type: str
    region: str
    
    # NODE 1 OUTPUT — IDP Extraction
    extracted_procedure: List[str]
    extracted_equipment: List[str]
    extracted_capability: List[str]
    extraction_status: str
    
    # NODE 2 OUTPUT — Anomaly Detection
    anomalies: List[str]
    anomaly_count: int
    trust_score: float        # 0.0 to 1.0 — how trustworthy are the claims?
    
    # NODE 3 OUTPUT — RAG Answer (filled later)
    ngo_question: str
    rag_answer: str
    supporting_hospitals: List[str]

print("✅ HospitalState defined!")
print("   This is the shared data that flows through all 3 nodes")
print("   Node 1 → extracts facts into the state")
print("   Node 2 → adds anomalies into the state")  
print("   Node 3 → adds NGO answer into the state")

# COMMAND ----------

# Cell 4 — Node 1: IDP Extractor
def idp_extractor_node(state: HospitalState) -> dict:
    """
    Node 1: Takes raw hospital description
    Extracts procedure, equipment, capability using Groq
    """
    print(f"  🤖 Node 1: Extracting facts for {state['hospital_name']}...")
    
    # Skip if description too short
    if len(str(state['description'])) < 50:
        return {
            "extracted_procedure": [],
            "extracted_equipment": [],
            "extracted_capability": [],
            "extraction_status": "SKIPPED_SHORT_DESC"
        }
    
    prompt = f"""
You are a medical facility information extractor.
Extract facts ONLY about this hospital: {state['hospital_name']}

Description:
{state['description']}

Return a JSON object with exactly these 3 fields:
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
        
        return {
            "extracted_procedure": result.get("procedure", []),
            "extracted_equipment": result.get("equipment", []),
            "extracted_capability": result.get("capability", []),
            "extraction_status": "SUCCESS"
        }
        
    except Exception as e:
        return {
            "extracted_procedure": [],
            "extracted_equipment": [],
            "extracted_capability": [],
            "extraction_status": f"ERROR: {str(e)[:50]}"
        }

print("✅ Node 1 ready: IDP Extractor")

# COMMAND ----------

# Cell 5 — Node 2: Anomaly Detector
def anomaly_detector_node(state: HospitalState) -> dict:
    """
    Node 2: Checks extracted facts for suspicious claims
    Gives each hospital a trust score
    """
    print(f"  🔍 Node 2: Checking anomalies for {state['hospital_name']}...")
    
    anomalies = []
    trust_score = 1.0  # Start at 100% trust, deduct for each anomaly
    
    facility_type = str(state.get('facility_type', '')).lower()
    all_text = ' '.join(
        state.get('extracted_procedure', []) +
        state.get('extracted_equipment', []) +
        state.get('extracted_capability', [])
    ).lower()
    
    # Anomaly 1: Clinic claiming ICU
    if facility_type == 'clinic' and 'icu' in all_text:
        anomalies.append('⚠️ Clinic claiming ICU — needs verification')
        trust_score -= 0.3
    
    # Anomaly 2: Pharmacy claiming surgery
    if facility_type == 'pharmacy' and 'surgery' in all_text:
        anomalies.append('⚠️ Pharmacy claiming surgery — highly suspicious')
        trust_score -= 0.5
    
    # Anomaly 3: Hospital with zero capabilities extracted
    if facility_type == 'hospital' and len(state.get('extracted_capability', [])) == 0:
        anomalies.append('⚠️ Hospital with no capabilities — incomplete data')
        trust_score -= 0.2
    
    # Anomaly 4: Claims ICU but no emergency care
    if 'icu' in all_text and 'emergency' not in all_text:
        anomalies.append('⚠️ Has ICU but no emergency care — inconsistent')
        trust_score -= 0.2
    
    # Anomaly 5: No region data
    if state.get('region', 'Unknown') == 'Unknown':
        anomalies.append('⚠️ No location data — cannot map to region')
        trust_score -= 0.1
    
    # Anomaly 6: Extraction failed
    if state.get('extraction_status', '').startswith('ERROR'):
        anomalies.append('⚠️ AI extraction failed — data unreliable')
        trust_score -= 0.4

    trust_score = round(max(0.0, trust_score), 2)
    
    return {
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "trust_score": trust_score
    }

print("✅ Node 2 ready: Anomaly Detector")

# COMMAND ----------

# Cell 6 — Node 3: RAG Planner (answers NGO questions)
def rag_planner_node(state: HospitalState) -> dict:
    """
    Node 3: If there's an NGO question, answer it using
    the extracted facts from this hospital
    """
    
    # If no question asked, skip
    if not state.get('ngo_question', ''):
        return {
            "rag_answer": "",
            "supporting_hospitals": []
        }
    
    print(f"  💬 Node 3: Answering NGO question...")
    
    # Build context from extracted facts
    facts = (
        state.get('extracted_procedure', []) +
        state.get('extracted_equipment', []) +
        state.get('extracted_capability', [])
    )
    
    if not facts:
        return {
            "rag_answer": "No facts available for this hospital.",
            "supporting_hospitals": []
        }
    
    facts_text = '\n'.join([f"- {f}" for f in facts])
    
    prompt = f"""
You are an NGO healthcare coordinator assistant.

Hospital: {state['hospital_name']}
Region: {state['region']}
Trust Score: {state['trust_score']} (1.0 = fully trusted)

Known Facts:
{facts_text}

NGO Question: {state['ngo_question']}

Answer the question based ONLY on the facts above.
If this hospital is relevant to the question, explain how.
If not relevant, say "Not relevant to this query."
Keep answer under 3 sentences.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "rag_answer": answer,
            "supporting_hospitals": [state['hospital_name']]
        }
        
    except Exception as e:
        return {
            "rag_answer": f"Error: {str(e)[:50]}",
            "supporting_hospitals": []
        }

print("✅ Node 3 ready: RAG Planner")

# COMMAND ----------

# Cell 7 — Build the LangGraph Pipeline
# This connects Node 1 → Node 2 → Node 3

def build_pipeline():
    
    # Create the graph
    graph = StateGraph(HospitalState)
    
    # Add the 3 nodes
    graph.add_node("idp_extractor",    idp_extractor_node)
    graph.add_node("anomaly_detector", anomaly_detector_node)
    graph.add_node("rag_planner",      rag_planner_node)
    
    # Connect them in order
    graph.set_entry_point("idp_extractor")
    graph.add_edge("idp_extractor",    "anomaly_detector")
    graph.add_edge("anomaly_detector", "rag_planner")
    graph.add_edge("rag_planner",      END)
    
    # Compile
    return graph.compile()

# Build it
pipeline = build_pipeline()

print("✅ LangGraph Pipeline Built!")
print()
print("   Flow:")
print("   idp_extractor → anomaly_detector → rag_planner → END")
print()
print("   Node 1: Extracts procedure/equipment/capability")
print("   Node 2: Checks for suspicious claims + trust score")
print("   Node 3: Answers NGO questions using extracted facts")

# COMMAND ----------

# Cell 8 — Test the full pipeline on 1 real hospital

# Load your enriched data
df = spark.table("enriched_facilities").toPandas()

# Pick a hospital with a good description
test_hospital = df[
    df['description'].notna() & 
    (df['description'].str.len() > 200)
].iloc[0]

print(f"🏥 Testing on: {test_hospital['name']}")
print(f"📍 Region: {test_hospital['region_clean']}")
print(f"📝 Description: {str(test_hospital['description'])[:200]}...")
print()
print("🚀 Running through LangGraph pipeline...")
print("=" * 60)

# Run the pipeline
result = pipeline.invoke({
    # Input
    "hospital_name":  test_hospital['name'],
    "description":    str(test_hospital['description']),
    "facility_type":  str(test_hospital['facilityTypeId']),
    "region":         str(test_hospital['region_clean']),
    
    # Node outputs (start empty)
    "extracted_procedure":  [],
    "extracted_equipment":  [],
    "extracted_capability": [],
    "extraction_status":    "",
    "anomalies":            [],
    "anomaly_count":        0,
    "trust_score":          1.0,
    
    # NGO question (optional - try one!)
    "ngo_question": "Does this hospital have emergency care or surgery?",
    "rag_answer":   "",
    "supporting_hospitals": []
})

# Show results
print(f"\n📋 PROCEDURES ({len(result['extracted_procedure'])}):")
for p in result['extracted_procedure']:
    print(f"   • {p}")

print(f"\n🔧 EQUIPMENT ({len(result['extracted_equipment'])}):")
for e in result['extracted_equipment']:
    print(f"   • {e}")

print(f"\n⚡ CAPABILITIES ({len(result['extracted_capability'])}):")
for c in result['extracted_capability']:
    print(f"   • {c}")

print(f"\n🔍 ANOMALIES ({result['anomaly_count']}):")
if result['anomalies']:
    for a in result['anomalies']:
        print(f"   {a}")
else:
    print("   ✅ No anomalies found")

print(f"\n🛡️  TRUST SCORE: {result['trust_score']} / 1.0")

print(f"\n💬 NGO ANSWER:")
print(f"   {result['rag_answer']}")

print(f"\n✅ Pipeline ran successfully!")

# COMMAND ----------

# Cell 9 — Test on a proper hospital with rich description

# Find hospitals (not NGOs) with detailed descriptions
proper_hospitals = df[
    df['description'].notna() &
    (df['description'].str.len() > 300) &
    (df['facilityTypeId'].str.lower() == 'hospital')
].copy()

print(f"Found {len(proper_hospitals)} proper hospitals with rich descriptions")
print()

# Test on top 3
for i in range(min(3, len(proper_hospitals))):
    hospital = proper_hospitals.iloc[i]
    
    print(f"{'='*60}")
    print(f"🏥 {hospital['name']}")
    print(f"📍 Region: {hospital['region_clean']}")
    print(f"📝 Description ({len(str(hospital['description']))} chars)")
    print()
    
    result = pipeline.invoke({
        "hospital_name":        hospital['name'],
        "description":          str(hospital['description']),
        "facility_type":        str(hospital['facilityTypeId']),
        "region":               str(hospital['region_clean']),
        "extracted_procedure":  [],
        "extracted_equipment":  [],
        "extracted_capability": [],
        "extraction_status":    "",
        "anomalies":            [],
        "anomaly_count":        0,
        "trust_score":          1.0,
        "ngo_question":         "Does this hospital have ICU, surgery or emergency care?",
        "rag_answer":           "",
        "supporting_hospitals": []
    })
    
    total_facts = (
        len(result['extracted_procedure']) +
        len(result['extracted_equipment']) +
        len(result['extracted_capability'])
    )
    
    print(f"  📋 Procedures  : {len(result['extracted_procedure'])}")
    print(f"  🔧 Equipment   : {len(result['extracted_equipment'])}")
    print(f"  ⚡ Capabilities: {len(result['extracted_capability'])}")
    print(f"  📊 Total facts : {total_facts}")
    print(f"  🛡️  Trust Score : {result['trust_score']}")
    print(f"  🔍 Anomalies   : {result['anomaly_count']}")
    print(f"  💬 NGO Answer  : {result['rag_answer'][:150]}...")
    print()
    
    time.sleep(1)

print("✅ Pipeline tested on real hospitals!")

# COMMAND ----------

# Cell 10 — Build FAISS Vector Store
# This converts all hospital descriptions into searchable embeddings

# MAGIC %pip install sentence-transformers faiss-cpu -q

# COMMAND ----------

# Cell 11 — Create embeddings for all hospitals

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("🔄 Loading embedding model...")
# This model converts text → numbers (embeddings)
# It runs locally, no API key needed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# Load all facilities with descriptions
df_with_desc = df[df['description'].notna() & 
                  (df['description'].str.len() > 50)].copy()

print(f"\n🏥 Hospitals to embed: {len(df_with_desc)}")
print("🔄 Creating embeddings (this takes 1-2 mins)...")

# Create rich text for each hospital
# Combine name + region + description + existing facts
def build_hospital_text(row):
    parts = [
        f"Hospital: {row['name']}",
        f"Region: {row['region_clean']}",
        f"Type: {row['facilityTypeId']}",
        f"Description: {str(row['description'])[:500]}",
    ]
    
    # Add existing facts if available
    if str(row.get('procedure', '')) not in ['', '[]', "['']", 'nan']:
        parts.append(f"Procedures: {row['procedure']}")
    if str(row.get('capability', '')) not in ['', '[]', "['']", 'nan']:
        parts.append(f"Capabilities: {row['capability']}")
    
    return ' | '.join(parts)

df_with_desc['search_text'] = df_with_desc.apply(build_hospital_text, axis=1)

# Create embeddings
texts = df_with_desc['search_text'].tolist()
embeddings = embedder.encode(texts, show_progress_bar=True)

print(f"\n✅ Embeddings created!")
print(f"   Shape: {embeddings.shape}")
print(f"   Each hospital = {embeddings.shape[1]} numbers")

# COMMAND ----------

# Cell 12 (FINAL) — Build HIGH-QUALITY searchable text for ALL hospitals

MAX_LEN = 800  # Prevent overly long text (important for embeddings)

def is_valid(val):
    """Check if a field has meaningful content"""
    return str(val) not in ['', 'nan', '[]', "['']", 'None']

def build_hospital_text_full(row):
    """
    Build natural-language searchable text for every hospital
    Optimized for semantic search (embedding-friendly)
    """
    parts = []
    
    # Core identity (always present)
    name = row.get('name', 'Unknown hospital')
    city = row.get('address_city', 'unknown city')
    region = row.get('region_clean', 'unknown region')
    
    parts.append(f"{name} is a healthcare facility located in {city}, {region}.")
    
    # Type
    if is_valid(row.get('facilityTypeId')):
        parts.append(f"It is a type {row['facilityTypeId']} hospital.")
    
    # Description
    if is_valid(row.get('description')):
        parts.append(f"About the hospital: {str(row['description'])[:400]}.")
    
    # Specialties (VERY IMPORTANT for search)
    if is_valid(row.get('specialties')):
        parts.append(f"It offers specialties such as {row['specialties']}.")
        parts.append(f"Key specialties include {row['specialties']}.")  # repetition boost
    
    # Procedures
    if is_valid(row.get('procedure')):
        parts.append(f"Procedures available include {row['procedure']}.")
    
    # Equipment
    if is_valid(row.get('equipment')):
        parts.append(f"The hospital is equipped with {row['equipment']}.")
    
    # Capabilities
    if is_valid(row.get('capability')):
        parts.append(f"It has capabilities such as {row['capability']}.")
    
    # Extracted insights (from IDP)
    if is_valid(row.get('extracted_procedure')):
        parts.append(f"Additional procedures include {row['extracted_procedure']}.")
    
    if is_valid(row.get('extracted_capability')):
        parts.append(f"Additional capabilities include {row['extracted_capability']}.")
    
    # Query-friendly sentence (boosts retrieval performance)
    if is_valid(row.get('specialties')):
        parts.append(f"This hospital is suitable for patients looking for {row['specialties']} treatments.")
    
    # Join + truncate
    full_text = " ".join(parts)
    return full_text[:MAX_LEN]


# =========================
# APPLY TO DATAFRAME
# =========================

print("🔄 Building search text for ALL hospitals...")

df['search_text_full'] = df.apply(build_hospital_text_full, axis=1)

# =========================
# VALIDATION
# =========================

has_good_text = df['search_text_full'].str.len() > 30
print(f"✅ Hospitals with searchable text: {has_good_text.sum()} / {len(df)}")

print(f"\n📊 Text length stats:")
print(f"   Average length : {df['search_text_full'].str.len().mean():.0f} chars")
print(f"   Min length     : {df['search_text_full'].str.len().min()} chars")
print(f"   Max length     : {df['search_text_full'].str.len().max()} chars")

# Sample check (no description case)
print(f"\n📋 Sample for a hospital with no description:")
no_desc_df = df[df['description'].isna()]

if len(no_desc_df) > 0:
    sample = no_desc_df.iloc[0]
    print(f"   {sample['search_text_full']}")
else:
    print("   No hospitals with missing description found.")

# COMMAND ----------

# Cell 13 — Embed ALL 987 hospitals into FAISS

print(f"🔄 Embedding ALL {len(df)} hospitals...")
print("⏰ This takes 2-3 minutes...")
print()

# Create embeddings for everyone
all_texts = df['search_text_full'].tolist()
all_embeddings = embedder.encode(
    all_texts, 
    show_progress_bar=True,
    batch_size=64  # Process 64 at a time for speed
)

print(f"\n✅ All embeddings created!")
print(f"   Shape: {all_embeddings.shape}")
print(f"   Total hospitals: {all_embeddings.shape[0]}")
print(f"   Embedding size : {all_embeddings.shape[1]} dimensions")

# Build FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings.astype('float32'))

print(f"\n✅ FAISS index built!")
print(f"   Hospitals indexed: {index.ntotal}")

# Save everything
faiss.write_index(index, "/tmp/ghana_hospitals_full.index")
df.to_pickle("/tmp/hospital_metadata_full.pkl")

print(f"\n✅ Saved:")
print(f"   /tmp/ghana_hospitals_full.index")
print(f"   /tmp/hospital_metadata_full.pkl")
print(f"\n🎉 ALL 987 hospitals are now searchable!")

# COMMAND ----------

# Cell 14 (FIXED) — Verify samples then add confidence scores

print("📋 Sample texts to verify quality:\n")

# Sample 1: Hospital WITH long description
with_desc = df[df['description'].notna() & 
               (df['description'].str.len() > 200)].iloc[0]
print("🟢 Hospital WITH rich description:")
print(f"   Name: {with_desc['name']}")
print(f"   Text: {with_desc['search_text_full'][:300]}...")
print()

# Sample 2: Hospital with SHORT description
short_desc = df[df['description'].str.len() < 100].iloc[0]
print("🟡 Hospital with SHORT description:")
print(f"   Name: {short_desc['name']}")
print(f"   Text: {short_desc['search_text_full']}")
print()

# Sample 3: Smallest text hospital
minimal = df.copy()
minimal['text_len'] = minimal['search_text_full'].str.len()
minimal = minimal.nsmallest(1, 'text_len').iloc[0]
print("🔴 Hospital with MINIMUM text:")
print(f"   Name: {minimal['name']}")
print(f"   Text: {minimal['search_text_full']}")
print()

# Add confidence score to each hospital
def get_confidence(row):
    has_desc = (
        str(row.get('description', '')) not in ['', 'nan', '[]'] and
        len(str(row.get('description', ''))) > 100
    )
    has_facts = str(row.get('capability', '')) not in ['', 'nan', '[]', "['']"]
    has_specialties = str(row.get('specialties', '')) not in ['', 'nan', '[]']
    
    if has_desc and has_facts:
        return 'High'
    elif has_desc or has_facts or has_specialties:
        return 'Medium'
    else:
        return 'Low'

df['confidence'] = df.apply(get_confidence, axis=1)

print("📊 CONFIDENCE BREAKDOWN:")
counts = df['confidence'].value_counts()
for level, count in counts.items():
    pct = round(count/len(df)*100, 1)
    bar = "█" * int(pct/3)
    print(f"  {level:<8} {bar} {count} hospitals ({pct}%)")

print(f"\n✅ Confidence labels added to all {len(df)} hospitals!")

# COMMAND ----------

# Cell 15 — Embed ALL 987 hospitals into FAISS

print(f"🔄 Embedding ALL {len(df)} hospitals...")
print("⏰ Takes 2-3 minutes...")
print()

all_texts = df['search_text_full'].tolist()
all_embeddings = embedder.encode(
    all_texts,
    show_progress_bar=True,
    batch_size=64
)

print(f"\n✅ Embeddings created!")
print(f"   Total    : {all_embeddings.shape[0]} hospitals")
print(f"   Dimensions: {all_embeddings.shape[1]}")

# Build FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings.astype('float32'))

# Save metadata with confidence
hospital_metadata_full = df[[
    'name', 'region_clean', 'facilityTypeId',
    'address_city', 'description',
    'procedure', 'equipment', 'capability',
    'specialties', 'search_text_full', 'confidence'
]].reset_index(drop=True)

faiss.write_index(index, "/tmp/ghana_hospitals_full.index")
hospital_metadata_full.to_pickle("/tmp/hospital_metadata_full.pkl")

print(f"\n✅ FAISS index saved!")
print(f"   Indexed: {index.ntotal} / 987 hospitals")
print()
print(f"📊 FINAL BREAKDOWN:")
print(f"  🟢 High confidence   : {(df['confidence'] == 'High').sum()}")
print(f"  🟡 Medium confidence : {(df['confidence'] == 'Medium').sum()}")
print(f"  🔴 Low confidence    : {(df['confidence'] == 'Low').sum()}")
print(f"\n🎉 ALL 987 hospitals searchable!")

# COMMAND ----------

# Cell 16 — Full RAG Search across ALL 987 hospitals

def rag_search_full(question, top_k=5):
    """
    Full RAG pipeline across all 987 hospitals:
    1. Embed the question
    2. FAISS finds top_k most relevant hospitals
    3. Groq answers using those hospitals as context
    4. Returns answer + citations
    """
    
    print(f"❓ Question: {question}")
    print(f"🔍 Searching across {index.ntotal} hospitals...")
    print()
    
    # Step 1: Embed the question
    question_embedding = embedder.encode([question]).astype('float32')
    
    # Step 2: Search FAISS
    distances, indices = index.search(question_embedding, top_k)
    
    # Step 3: Get matching hospitals
    relevant_hospitals = []
    print(f"📍 Top {top_k} most relevant hospitals:")
    print("-" * 50)
    
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        hospital = hospital_metadata_full.iloc[idx]
        relevant_hospitals.append(hospital)
        similarity = round(1 / (1 + dist), 3)
        confidence = hospital['confidence']
        emoji = "🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"
        
        print(f"  {rank+1}. {hospital['name']}")
        print(f"     Region     : {hospital['region_clean']}")
        print(f"     Confidence : {emoji} {confidence}")
        print(f"     Similarity : {similarity}")
        print()
    
    # Step 4: Build context for Groq
    context = ""
    for i, h in enumerate(relevant_hospitals):
        context += f"""
Hospital {i+1}: {h['name']}
Region: {h['region_clean']}
Type: {h['facilityTypeId']}
Confidence: {h['confidence']}
Procedures: {h['procedure']}
Equipment: {h['equipment']}
Capabilities: {h['capability']}
Specialties: {h['specialties']}
---"""
    
    # Step 5: Groq answers using context
    prompt = f"""
You are an NGO healthcare coordinator assistant for Ghana.
You help NGO workers find the right hospitals for patients.

Based on these {top_k} most relevant hospitals found by search:
{context}

Answer this question: {question}

Instructions:
- Be specific about WHICH hospitals can help and their REGION
- Mention confidence level of each hospital's data
- If none match well, say so clearly
- Flag any Low confidence results
- End your answer with:
  "📌 Sources: [list hospital names used]"
- Keep answer under 200 words
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    answer = response.choices[0].message.content.strip()
    
    print(f"💬 ANSWER:")
    print(f"{answer}")
    print()
    
    return {
        "question": question,
        "answer": answer,
        "hospitals_used": [h['name'] for h in relevant_hospitals],
        "confidences": [h['confidence'] for h in relevant_hospitals]
    }

print("✅ Full RAG search ready!")
print(f"   Searching across: {index.ntotal} hospitals")

# COMMAND ----------

# Cell 17 — Test with 4 real NGO questions

questions = [
    "Which hospitals in Northern Ghana have emergency care?",
    "Where can I find surgery facilities in Volta region?",
    "Which hospitals offer maternity and pediatric care together?",
    "What hospitals are available in Upper East region?"
]

all_results = []

for q in questions:
    print("=" * 65)
    result = rag_search_full(q, top_k=5)
    all_results.append(result)
    print()
    time.sleep(1)

print("=" * 65)
print("✅ RAG TEST COMPLETE!")
print(f"\n📊 SUMMARY:")
print(f"   Questions answered : {len(all_results)}")
print(f"   Hospitals searched : {index.ntotal} per question")
print(f"   Model used         : llama-3.3-70b-versatile (Groq)")
print(f"\n🎉 Your full RAG system works across all 987 hospitals!")

# COMMAND ----------

# Cell 18 — Improved RAG with region pre-filtering

def rag_search_v2(question, top_k=5, region_filter=None):
    """
    Improved RAG:
    - If region mentioned in question, filter to that region first
    - Then do FAISS similarity search within that subset
    - Falls back to full search if region has too few results
    """
    
    print(f"❓ Question: {question}")
    
    # --- Step 1: Auto-detect region from question ---
    ghana_regions = {
        'northern': 'Northern',
        'upper east': 'Upper East', 
        'upper west': 'Upper West',
        'greater accra': 'Greater Accra',
        'accra': 'Greater Accra',
        'ashanti': 'Ashanti',
        'kumasi': 'Ashanti',
        'volta': 'Volta',
        'western': 'Western',
        'central': 'Central',
        'eastern': 'Eastern',
        'brong ahafo': 'Brong Ahafo',
        'savannah': 'Savannah',
        'north east': 'North East',
        'oti': 'Oti',
        'ahafo': 'Ahafo',
        'bono east': 'Bono East',
        'western north': 'Western North',
    }
    
    detected_region = region_filter
    question_lower = question.lower()
    
    if not detected_region:
        for keyword, region in ghana_regions.items():
            if keyword in question_lower:
                detected_region = region
                break
    
    # --- Step 2: Filter hospitals by region if detected ---
    if detected_region:
        print(f"📍 Region detected: {detected_region}")
        region_df = hospital_metadata_full[
            hospital_metadata_full['region_clean'] == detected_region
        ]
        print(f"🏥 Hospitals in {detected_region}: {len(region_df)}")
        
        if len(region_df) >= 3:
            # Search within region only
            region_indices = region_df.index.tolist()
            region_embeddings = all_embeddings[region_indices].astype('float32')
            
            # Build mini FAISS index for this region
            mini_index = faiss.IndexFlatL2(dimension)
            mini_index.add(region_embeddings)
            
            # Search
            question_embedding = embedder.encode([question]).astype('float32')
            distances, local_indices = mini_index.search(
                question_embedding, 
                min(top_k, len(region_df))
            )
            
            # Map back to original indices
            actual_indices = [region_indices[i] for i in local_indices[0]]
            actual_distances = distances[0]
            search_scope = f"{detected_region} region only"
            
        else:
            # Too few hospitals in region — fall back to full search
            print(f"⚠️  Only {len(region_df)} hospitals in {detected_region}")
            print(f"   Falling back to full search...")
            question_embedding = embedder.encode([question]).astype('float32')
            actual_distances, actual_indices_arr = index.search(question_embedding, top_k)
            actual_indices = actual_indices_arr[0]
            actual_distances = actual_distances[0]
            search_scope = "all Ghana (region too small)"
    else:
        # No region detected — search everything
        print(f"🌍 No region detected — searching all Ghana")
        question_embedding = embedder.encode([question]).astype('float32')
        actual_distances, actual_indices_arr = index.search(question_embedding, top_k)
        actual_indices = actual_indices_arr[0]
        actual_distances = actual_distances[0]
        search_scope = "all Ghana"
    
    print(f"🔍 Search scope: {search_scope}")
    print()
    
    # --- Step 3: Get matching hospitals ---
    relevant_hospitals = []
    print(f"📍 Top results:")
    print("-" * 55)
    
    for rank, (dist, idx) in enumerate(zip(actual_distances, actual_indices)):
        hospital = hospital_metadata_full.iloc[idx]
        relevant_hospitals.append(hospital)
        similarity = round(1 / (1 + dist), 3)
        emoji = "🟢" if hospital['confidence'] == "High" else \
                "🟡" if hospital['confidence'] == "Medium" else "🔴"
        
        print(f"  {rank+1}. {hospital['name']}")
        print(f"     Region     : {hospital['region_clean']}")
        print(f"     Confidence : {emoji} {hospital['confidence']}")
        print(f"     Similarity : {similarity}")
        print()
    
    # --- Step 4: Build context ---
    context = ""
    for i, h in enumerate(relevant_hospitals):
        context += f"""
Hospital {i+1}: {h['name']}
Region: {h['region_clean']}
Type: {h['facilityTypeId']}
Confidence: {h['confidence']}
Procedures: {h['procedure']}
Equipment: {h['equipment']}
Capabilities: {h['capability']}
Specialties: {h['specialties']}
---"""
    
    # --- Step 5: Groq answers ---
    prompt = f"""
You are an NGO healthcare coordinator assistant for Ghana.
You help NGO workers find the right hospitals for patients.

Search scope: {search_scope}
Hospitals found: {len(relevant_hospitals)}

Hospital data:
{context}

Question: {question}

Instructions:
- Be specific about WHICH hospitals can help and their REGION
- If data confidence is Low or Medium, mention to verify first
- If a region has very few hospitals, flag it as a potential medical desert
- If none match, clearly state that and suggest nearest alternatives
- End with: "📌 Sources: [hospital names used]"
- Keep answer under 200 words
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    answer = response.choices[0].message.content.strip()
    
    print(f"💬 ANSWER:")
    print(answer)
    print()
    
    return {
        "question": question,
        "answer": answer,
        "region_detected": detected_region,
        "search_scope": search_scope,
        "hospitals_used": [h['name'] for h in relevant_hospitals],
        "confidences": [h['confidence'] for h in relevant_hospitals]
    }

print("✅ Improved RAG v2 ready!")
print("   Now includes automatic region detection + filtering!")

# COMMAND ----------

# Cell 19 — Retest with improved RAG v2

questions = [
    "Which hospitals in Northern Ghana have emergency care?",
    "Where can I find surgery facilities in Volta region?",
    "Which hospitals offer maternity and pediatric care together?",
    "What hospitals are available in Upper East region?"
]

all_results = []

for q in questions:
    print("=" * 65)
    result = rag_search_v2(q, top_k=5)
    all_results.append(result)
    print()
    time.sleep(1)

print("=" * 65)
print("✅ IMPROVED RAG TEST COMPLETE!")
print()
print("📊 REGION DETECTION SUMMARY:")
for r in all_results:
    detected = r['region_detected'] or 'None (global search)'
    print(f"  Q: {r['question'][:45]}...")
    print(f"     Region detected : {detected}")
    print(f"     Scope           : {r['search_scope']}")
    print()

# COMMAND ----------

# Cell 20 — Save everything to Databricks permanently

import mlflow
import pickle
import numpy as np

print("💾 Saving all progress...\n")

# ── 1. Save enriched dataframe as Delta Table ──────────────────
spark_df = spark.createDataFrame(
    hospital_metadata_full.astype(str).fillna('')
)
spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hospital_metadata_full")

print("✅ 1/5 Delta table saved: hospital_metadata_full")

# ── 2. Save FAISS index to Databricks Volume ───────────────────
faiss.write_index(index, "/tmp/ghana_hospitals_full.index")

# Copy to persistent Databricks storage
dbutils.fs.cp(
    "file:/tmp/ghana_hospitals_full.index",
    "/Volumes/workspace/default/project/ghana_hospitals_full.index"
)
print("✅ 2/5 FAISS index saved to Volume")

# ── 3. Save hospital metadata pickle ──────────────────────────
hospital_metadata_full.to_pickle("/tmp/hospital_metadata_full.pkl")

dbutils.fs.cp(
    "file:/tmp/hospital_metadata_full.pkl",
    "/Volumes/workspace/default/project/hospital_metadata_full.pkl"
)
print("✅ 3/5 Hospital metadata saved to Volume")

# ── 4. Save embeddings ─────────────────────────────────────────
np.save("/tmp/all_embeddings.npy", all_embeddings)

dbutils.fs.cp(
    "file:/tmp/all_embeddings.npy",
    "/Volumes/workspace/default/project/all_embeddings.npy"
)
print("✅ 4/5 Embeddings saved to Volume")

# ── 5. Log everything to MLflow ────────────────────────────────
mlflow.set_experiment(
    "/Users/cp4707@srmist.edu.in/medical-desert-idp-agent"
)

with mlflow.start_run(run_name="Ghana_RAG_Pipeline_v2"):
    
    # Params
    mlflow.log_param("ai_model",        "llama-3.3-70b-versatile")
    mlflow.log_param("ai_provider",     "Groq")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("vector_store",    "FAISS")
    mlflow.log_param("orchestration",   "LangGraph")
    mlflow.log_param("prompt_version",  "v2")
    mlflow.log_param("country",         "Ghana")
    
    # Metrics
    mlflow.log_metric("total_facilities",     987)
    mlflow.log_metric("facilities_indexed",   index.ntotal)
    mlflow.log_metric("high_confidence",      
                      (hospital_metadata_full['confidence'] == 'High').sum())
    mlflow.log_metric("medium_confidence",    
                      (hospital_metadata_full['confidence'] == 'Medium').sum())
    mlflow.log_metric("low_confidence",       
                      (hospital_metadata_full['confidence'] == 'Low').sum())
    mlflow.log_metric("embedding_dimensions", 384)
    mlflow.log_metric("langgraph_nodes",      3)
    
    # Log FAISS index as artifact
    mlflow.log_artifact("/tmp/ghana_hospitals_full.index")
    mlflow.log_artifact("/tmp/hospital_metadata_full.pkl")
    
    print("✅ 5/5 MLflow experiment logged")

print("\n" + "="*50)
print("🎉 ALL PROGRESS SAVED!")
print("="*50)
print(f"""
📦 What was saved:
   Delta Table  → hospital_metadata_full  (987 rows)
   FAISS Index  → ghana_hospitals_full.index
   Metadata     → hospital_metadata_full.pkl
   Embeddings   → all_embeddings.npy
   MLflow Run   → Ghana_RAG_Pipeline_v2

🏗️  Architecture saved:
   ✅ LangGraph  (3 nodes)
   ✅ FAISS RAG  (987 hospitals)
   ✅ Groq LLM   (llama-3.3-70b)
   ✅ MLflow     (experiment tracked)
""")

# COMMAND ----------

# Cell 20 (FIXED) — Save everything using correct Databricks Serverless approach

import mlflow
import numpy as np

print("💾 Saving all progress...\n")

# ── 1. Delta table already saved ✅ ───────────────────────────
print("✅ 1/4 Delta table: hospital_metadata_full (already saved)")

# ── 2. Save FAISS index as binary directly to Volume ──────────
try:
    # Write FAISS index to bytes then save via spark
    faiss.write_index(index, "/tmp/ghana_hospitals_full.index")
    
    # Read and save as spark binary file to Volume
    with open("/tmp/ghana_hospitals_full.index", "rb") as f:
        index_bytes = f.read()
    
    # Save via pandas to Delta table (binary storage trick)
    import pandas as pd
    binary_df = pd.DataFrame({
        'filename': ['ghana_hospitals_full.index'],
        'data': [index_bytes.hex()]  # Store as hex string
    })
    spark.createDataFrame(binary_df) \
         .write.format("delta") \
         .mode("overwrite") \
         .option("overwriteSchema", "true") \
         .saveAsTable("faiss_index_storage")
    
    print("✅ 2/4 FAISS index saved to Delta table: faiss_index_storage")
except Exception as e:
    print(f"⚠️  FAISS save issue: {str(e)[:80]}")
    print("   (Index is still in memory — will rebuild if needed)")

# ── 3. Save metadata as Delta table ───────────────────────────
try:
    spark.createDataFrame(
        hospital_metadata_full.astype(str).fillna('')
    ).write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("hospital_metadata_full")
    print("✅ 3/4 Metadata saved to Delta table: hospital_metadata_full")
except Exception as e:
    print(f"⚠️  Metadata save issue: {str(e)[:80]}")

# ── 4. Log to MLflow ──────────────────────────────────────────
try:
    mlflow.set_experiment(
        "/Users/cp4707@srmist.edu.in/medical-desert-idp-agent"
    )
    
    with mlflow.start_run(run_name="Ghana_RAG_Pipeline_v2"):
        
        # Params
        mlflow.log_param("ai_model",        "llama-3.3-70b-versatile")
        mlflow.log_param("ai_provider",     "Groq")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("vector_store",    "FAISS")
        mlflow.log_param("orchestration",   "LangGraph")
        mlflow.log_param("prompt_version",  "v2")
        mlflow.log_param("country",         "Ghana")
        
        # Metrics
        mlflow.log_metric("total_facilities",     987)
        mlflow.log_metric("facilities_indexed",   index.ntotal)
        mlflow.log_metric("high_confidence",
            int((hospital_metadata_full['confidence'] == 'High').sum()))
        mlflow.log_metric("medium_confidence",
            int((hospital_metadata_full['confidence'] == 'Medium').sum()))
        mlflow.log_metric("low_confidence",
            int((hospital_metadata_full['confidence'] == 'Low').sum()))
        mlflow.log_metric("embedding_dimensions", 384)
        mlflow.log_metric("langgraph_nodes",      3)
        mlflow.log_metric("rag_search_regions",   16)
        
        # Log FAISS index as artifact from /tmp
        mlflow.log_artifact("/tmp/ghana_hospitals_full.index")
        print("✅ 4/4 MLflow experiment logged: Ghana_RAG_Pipeline_v2")
        
except Exception as e:
    print(f"⚠️  MLflow issue: {str(e)[:80]}")

print("\n" + "="*50)
print("🎉 PROGRESS SAVED!")
print("="*50)
print(f"""
📦 What's saved:
   ✅ Delta Table → hospital_metadata_full  (987 rows)
   ✅ Delta Table → faiss_index_storage
   ✅ Delta Table → enriched_facilities
   ✅ Delta Table → region_gap_analysis
   ✅ Delta Table → facility_anomalies
   ✅ MLflow     → Ghana_RAG_Pipeline_v2

🏗️  Architecture built today:
   ✅ LangGraph  → 3 nodes (Extract→Detect→Answer)
   ✅ FAISS RAG  → 987 hospitals indexed
   ✅ Groq LLM   → llama-3.3-70b-versatile
   ✅ Region     → auto-detection in queries
   ✅ Confidence → High/Medium/Low scoring
   ✅ Citations  → every RAG answer sourced

⬜ Still to build:
   → Streamlit UI
   → Deploy online
   → README + demo video
""")

# COMMAND ----------

# Export everything needed for Streamlit Cloud

# 1. Export metadata CSV
hospital_metadata_full.to_csv(
    "/tmp/hospital_metadata.csv", 
    index=False
)
print(f"✅ CSV exported: {len(hospital_metadata_full)} rows")

# 2. Check map exists
import os
if os.path.exists("/tmp/ghana_map.html"):
    print("✅ Map exists at /tmp/ghana_map.html")
else:
    print("❌ Map not found - run Notebook 3 Cell 8 first")

# 3. Show file sizes
import os
csv_size = os.path.getsize("/tmp/hospital_metadata.csv")
print(f"   CSV size: {csv_size/1024:.1f} KB")


# COMMAND ----------

# Reload everything from Delta tables
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("🔄 Reloading data from Delta tables...")

# Load metadata
hospital_metadata_full = spark.table(
    "hospital_metadata_full"
).toPandas()
print(f"✅ Loaded: {len(hospital_metadata_full)} hospitals")

# Rebuild embeddings + FAISS index
print("🔄 Rebuilding embeddings (2-3 mins)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_text(row):
    parts = [
        f"Hospital: {row.get('name','')}",
        f"Region: {row.get('region_clean','')}",
        f"Type: {row.get('facilityTypeId','')}",
    ]
    for col in ['description','specialties',
                'procedure','capability','equipment']:
        val = str(row.get(col,''))
        if val not in ['','nan','[]',"['']"]:
            parts.append(f"{col}: {val[:300]}")
    return ' | '.join(parts)

hospital_metadata_full['search_text_full'] = (
    hospital_metadata_full.apply(build_text, axis=1)
)

all_embeddings = embedder.encode(
    hospital_metadata_full['search_text_full'].tolist(),
    show_progress_bar=True,
    batch_size=64
)

dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings.astype('float32'))

print(f"✅ FAISS index rebuilt: {index.ntotal} hospitals")
print(f"✅ All variables ready!")

# COMMAND ----------

# Export for Streamlit Cloud
hospital_metadata_full.to_csv(
    "/tmp/hospital_metadata.csv",
    index=False
)

import os
csv_size = os.path.getsize("/tmp/hospital_metadata.csv")
print(f"✅ CSV exported: {len(hospital_metadata_full)} rows")
print(f"   File size: {csv_size/1024:.1f} KB")

if os.path.exists("/tmp/ghana_map.html"):
    print("✅ Map exists!")
else:
    print("❌ Map missing — run Notebook 3 Cell 8 first")

# COMMAND ----------

# Regenerate the map
# MAGIC %pip install folium -q

# COMMAND ----------

import folium
import pandas as pd

# Load region gap analysis
region_data = spark.table("region_gap_analysis").toPandas()

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
<h3 align="center" style="font-size:18px;color:#333;margin-top:10px">
    🏥 Ghana Medical Desert Map — Virtue Foundation IDP Agent
</h3>
'''
ghana_map.get_root().html.add_child(
    folium.Element(title_html)
)

plotted = 0
for _, row in region_data.iterrows():
    region = row['region_clean']
    if region not in region_coords:
        continue

    lat, lon = region_coords[region]
    risk = row['risk_level']
    color = color_map.get(risk, 'gray')

    missing = []
    for svc, col in [
        ('ICU','has_icu'),
        ('Emergency','has_emergency'),
        ('Surgery','has_surgery'),
        ('Maternity','has_maternity'),
        ('Lab','has_lab'),
        ('Imaging','has_imaging'),
        ('Pediatrics','has_pediatrics'),
        ('Pharmacy','has_pharmacy'),
    ]:
        if str(row.get(col, '0')) == '0':
            missing.append(svc)

    popup_html = f"""
    <div style="width:220px;font-family:Arial">
        <h4 style="color:#333">{region}</h4>
        <hr>
        <b>Risk:</b> {risk}<br>
        <b>Facilities:</b> {row['total_facilities']}<br>
        <b>Services:</b> {row['services_available']}/8<br>
        <hr>
        <b>ICU:</b> 
        {'✅' if str(row.get('has_icu','0'))!='0' 
         else '❌'}&nbsp;
        <b>ER:</b> 
        {'✅' if str(row.get('has_emergency','0'))!='0' 
         else '❌'}&nbsp;
        <b>Surgery:</b> 
        {'✅' if str(row.get('has_surgery','0'))!='0' 
         else '❌'}<br>
        <b>Lab:</b> 
        {'✅' if str(row.get('has_lab','0'))!='0' 
         else '❌'}&nbsp;
        <b>Imaging:</b> 
        {'✅' if str(row.get('has_imaging','0'))!='0' 
         else '❌'}<br>
        <hr>
        <b style="color:red">
        Missing: {', '.join(missing) if missing else 'None ✅'}
        </b>
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
        tooltip=(
            f"📍 {region} | {risk} | "
            f"{row['total_facilities']} facilities"
        )
    ).add_to(ghana_map)
    plotted += 1

# Legend
legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;
     z-index:1000;background:white;padding:15px;
     border-radius:10px;border:2px solid #ccc;
     font-size:13px;font-family:Arial">
    <b>🏥 Medical Desert Risk</b><br><br>
    <span style="color:red;font-size:18px">●</span> 
    Critical Desert<br>
    <span style="color:orange;font-size:18px">●</span> 
    High Risk<br>
    <span style="color:#cccc00;font-size:18px">●</span> 
    Moderate Risk<br>
    <span style="color:green;font-size:18px">●</span> 
    Adequate Coverage<br>
    <br>
    <small>Circle size = number of facilities</small>
</div>
'''
ghana_map.get_root().html.add_child(
    folium.Element(legend_html)
)

ghana_map.save('/tmp/ghana_map.html')
print(f"✅ Map created! {plotted} regions plotted!")
print("✅ Saved to /tmp/ghana_map.html")
ghana_map

# COMMAND ----------

import os

files = [
    "/tmp/hospital_metadata.csv",
    "/tmp/ghana_map.html"
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"✅ {f} — {size:.1f} KB")
    else:
        print(f"❌ {f} — NOT FOUND")

# COMMAND ----------

# Copy files to your Databricks Volume
# (this is the downloadable area)

dbutils.notebook.entry_point.getDbutils().notebook()\
    .getContext().apiToken().get()

# Save CSV to Volume
import shutil

# Method 1 - try direct volume write
try:
    shutil.copy(
        "/tmp/hospital_metadata.csv",
        "/Volumes/workspace/default/project/hospital_metadata.csv"
    )
    print("✅ CSV copied to Volume")
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    # Method 2 - use spark to write
    spark.createDataFrame(
        hospital_metadata_full.astype(str).fillna('')
    ).coalesce(1).write.mode("overwrite").csv(
        "/Volumes/workspace/default/project/metadata_export",
        header=True
    )
    print("✅ CSV written via Spark to Volume")

# Save map to Volume  
try:
    with open("/tmp/ghana_map.html", "r") as f:
        map_content = f.read()
    
    with open(
        "/Volumes/workspace/default/project/ghana_map.html", 
        "w"
    ) as f:
        f.write(map_content)
    print("✅ Map copied to Volume")
except Exception as e:
    print(f"Map copy failed: {e}")

# COMMAND ----------

# ==============================
# SETUP (Run once)
# ==============================

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

client = Groq(api_key="YOUR_API_KEY")

# Load data
hospital_metadata_full = spark.table("hospital_metadata_full").toPandas()
hospital_metadata_full = hospital_metadata_full.fillna("")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Build embeddings (only once)
texts = hospital_metadata_full['search_text_full'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} hospitals")


# ==============================
# RAG FUNCTION
# ==============================

def ask_question_rag(question, k=20):
    print(f"\n{'='*60}")
    print(f"❓ QUESTION: {question}")
    print('='*60)
    
    # Step 1: Convert question → embedding
    query_embedding = model.encode([question]).astype("float32")
    
    # Step 2: FAISS search
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Get top results
    matches = hospital_metadata_full.iloc[indices[0]]
    
    # Step 4: Build context
    context = ""
    for i, (_, row) in enumerate(matches.iterrows(), 1):
        context += f"""
Hospital {i}: {row.get('name', 'Unknown')}
- Region: {row.get('region_clean', 'Unknown')}
- City: {row.get('address_city', 'Unknown')}
- Type: {row.get('facilityTypeId', 'Unknown')}
- Specialties: {str(row.get('specialties',''))[:150]}
- Capability: {str(row.get('capability',''))[:150]}
- Procedure: {str(row.get('procedure',''))[:150]}
---"""
    
    # Step 5: Prompt
    prompt = f"""You are an AI assistant helping NGO coordinators 
at the Virtue Foundation in Ghana.

Use ONLY the hospital data below to answer.

Hospital Data:
{context}

Question: {question}

Give:
1. Clear answer
2. Key hospitals
3. Recommendations
4. Flag anomalies if any
"""
    
    # Step 6: LLM call
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    
    answer = response.choices[0].message.content
    
    # Step 7: Print result
    print(f"\n🤖 AI ANSWER:\n{answer}")
    print(f"\n📋 BASED ON {len(matches)} HOSPITALS:")
    
    for _, row in matches.iterrows():
        print(f"   • {row.get('name')} — {row.get('region_clean')} — {row.get('facilityTypeId')}")

# COMMAND ----------

# ==============================
# SETUP
# ==============================
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 🔐 API Key
client = Groq(api_key=os.environ.get("GROQ_KEY"))

# ==============================
# LOAD DATA
# ==============================
hospital_metadata_full = spark.table("hospital_metadata_full").toPandas()
hospital_metadata_full = hospital_metadata_full.fillna("")

print(f"✅ Loaded {len(hospital_metadata_full)} hospitals")

# ==============================
# LOAD EMBEDDING MODEL
# ==============================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==============================
# BUILD FAISS INDEX
# ==============================
print("🔄 Creating embeddings...")

texts = hospital_metadata_full['search_text_full'].tolist()

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} hospitals")

# ==============================
# RAG FUNCTION
# ==============================
def ask_question_rag(question, k=20):
    print("\n" + "="*60)
    print(f"❓ QUESTION: {question}")
    print("="*60)
    
    # Step 1: Question → embedding
    query_embedding = model.encode([question]).astype("float32")
    
    # Step 2: Search FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Get results
    matches = hospital_metadata_full.iloc[indices[0]]
    
    # Step 4: Build context
    context = ""
    for i, (_, row) in enumerate(matches.iterrows(), 1):
        context += f"""
Hospital {i}: {row.get('name', 'Unknown')}
- Region: {row.get('region_clean', 'Unknown')}
- City: {row.get('address_city', 'Unknown')}
- Type: {row.get('facilityTypeId', 'Unknown')}
- Specialties: {str(row.get('specialties',''))[:150]}
- Capability: {str(row.get('capability',''))[:150]}
- Procedure: {str(row.get('procedure',''))[:150]}
---"""
    
    # Step 5: Prompt
    prompt = f"""You are an AI assistant helping NGO coordinators 
at the Virtue Foundation in Ghana.

Use ONLY the hospital data below.

Hospital Data:
{context}

Question: {question}

Give:
1. Clear answer
2. Key hospitals
3. Recommendations
4. Flag anomalies if any
"""
    
    # Step 6: LLM call
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    
    answer = response.choices[0].message.content
    
    # Step 7: Print
    print(f"\n🤖 AI ANSWER:\n{answer}")
    print(f"\n📋 BASED ON {len(matches)} HOSPITALS:")
    
    for _, row in matches.iterrows():
        print(f"   • {row.get('name')} — {row.get('region_clean')} — {row.get('facilityTypeId')}")


# ==============================
# TEST QUESTIONS (FINAL)
# ==============================

ask_question_rag("Which hospitals are in Accra?")
ask_question_rag("Which hospitals have emergency care?")
ask_question_rag("Which regions lack ICU facilities?")
ask_question_rag("Find hospitals with surgery in Ashanti region")
ask_question_rag("Which areas are medical deserts in Ghana?")

# COMMAND ----------

ask_question_rag("Which hospitals are in Accra?")
ask_question_rag("Which hospitals have emergency care?")
ask_question_rag("Which regions lack ICU facilities?")
ask_question_rag("Find hospitals with surgery in Ashanti region")
ask_question_rag("Which areas are medical deserts in Ghana?")
