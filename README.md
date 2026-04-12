# ⚕️ Ghana Medical Desert Agent
### *Identifying Healthcare Gaps · Routing NGO Resources · Saving Lives*

> **Hackathon project built for the Virtue Foundation** — an AI-powered intelligence platform that maps Ghana's healthcare coverage, identifies medical deserts, and helps NGO coordinators route resources where they're needed most.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/Search-FAISS%20%2B%20BM25-blue)](https://faiss.ai)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## 🌟 Live Demo & Workflow Video

- **Workflow / Demo Video:** [Watch the Video (working.mp4)](https://drive.google.com/file/d/1_p0fQsiERITjyMZidQwb1knKzIeR97Kw/view?usp=sharing)
- **Live Streamlit App:** [Insert Streamlit Cloud URL Here](https://share.streamlit.io/)

---

## 🌍 The Problem: Ghana's Medical Deserts

A **medical desert** is a region where people lack adequate access to basic healthcare — no emergency care, no surgery, no ICU, no maternity services. In Ghana, this is a daily reality for millions:

- **16 administrative regions**, but healthcare is heavily concentrated in Greater Accra and Ashanti
- Rural regions like **North East, Oti, Savannah, and Upper West** have critically few facilities
- NGO workers and relief coordinators struggle to find answers to basic questions:
  - *"Where is the nearest hospital with ICU capabilities?"*
  - *"Which region should we deploy to first?"*
  - *"How many facilities in Northern Ghana have emergency care?"*

This project answers all of those questions — instantly, accurately, and from a single interface.

---

## 🤖 What This Agent Does

The **Ghana Medical Desert Agent** is a full-stack AI application that:

1. **Indexes 896 real Ghana healthcare facilities** from the Virtue Foundation dataset
2. **Extracts structured capabilities** (procedures, equipment, specialties) from raw descriptions using a **Pydantic-validated** LLM-powered IDP pipeline, complete with source citations.
3. **Runs hybrid semantic + keyword search** (BM25 + FAISS + Cross-Encoder reranking) to find the most relevant facilities for any natural language query.
4. **Routes questions intelligently** — counting questions bypass the vector store entirely and query all 896 rows for accuracy.
5. **Generates AI-powered answers** using Groq's LLaMA 3.3 70B model with full dataset statistics as context.
6. **Visualises gaps on an interactive map** with risk-coded regional overlays.
7. **Emergency Patient Routing & Doctor Deployment** — computes Haversine distances to route patients to the nearest capable hospital, and identifies surplus regions to send doctors to critical deserts.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                           │
│  Dashboard | Search | Regional Analysis | Map | Directory        │
└──────────────────┬───────────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Question Router   │  ← Detects query type BEFORE search
        │  count / anomaly /  │
        │  gap / search       │
        └──────────┬──────────┘
          ┌────────┴────────────────────────────┐
          │                                     │
┌─────────▼──────────┐              ┌───────────▼────────────┐
│  Pre-Computed Stats │              │   Hybrid Search Engine  │
│  (all 896 rows)     │              │                        │
│  ─ service counts   │              │  BM25 (keyword/lexical)│
│  ─ by-region counts │              │       +                │
│  ─ anomaly flags    │              │  FAISS (semantic/dense)│
│  ─ gap analysis     │              │  all-MiniLM-L6-v2      │
└─────────┬──────────┘              └───────────┬────────────┘
          │                                     │
          └──────────────┬──────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Groq LLaMA 3.3    │  ← Receives BOTH stats + examples
              │   70B Versatile     │     to prevent hallucination
              └─────────────────────┘
```

### Why This Architecture?

Most RAG systems make a critical mistake: they pass only the top-5 retrieved results to the LLM. If you ask *"how many hospitals have ICU?"*, the LLM sees 5 results and says "5" — which is completely wrong.

Our **Question Router** solves this:
- `COUNT` questions → query pre-computed stats across all 896 rows → accurate answer
- `ANOMALY` questions → scan pre-computed anomaly flags → no hallucination
- `GAP` questions → regional gap analysis table → full picture
- `SEARCH` questions → hybrid BM25+FAISS → top-8 results + Groq with stats context

---

## 🔬 Data Pipeline (Databricks Notebooks)

The data powering this app was built through a 5-stage pipeline on Databricks:

### `01_explore_and_clean.py` — Data Exploration & Cleaning
- Loads the raw Virtue Foundation Ghana CSV (~987 rows)
- Assesses data quality: fill rates for name, region, description, procedure, equipment, capability
- Normalises messy region names: `"Greater Accra Region"`, `"ACCRA"`, `"Ga East"` → `"Greater Accra"`
- Flags each row with quality issues: `NEEDS_IDP_EXTRACTION`, `NO_LOCATION`, `NO_FACILITY_TYPE`
- Identifies initial medical desert candidates by region
- Saves cleaned data to a Delta Lake table: `medical_facilities_clean`

### `02_idp_agent.py` — IDP (Intelligent Document Processing) Agent
- **This is the core AI enrichment step**
- Many hospital records have rich free-text descriptions but no structured `procedure`, `equipment`, or `capability` data
- Uses Groq's **LLaMA 3.3 70B** to extract structured facts from each description:
  ```json
  {
    "procedure": ["Performs emergency surgery", "Offers antenatal care"],
    "equipment": ["Has X-ray machine", "Has operating theatre"],
    "capability": ["Provides 24/7 emergency care", "Has ICU unit"]
  }
  ```
- Processes ~440 hospitals that needed extraction
- Also uses AI to resolve unknown region assignments from hospital descriptions and addresses
- Saves enriched data to Delta table: `enriched_facilities`

### `03_gap_analysis.py` — Regional Gap Analysis & Anomaly Detection
- Scores each region on 8 critical service categories:
  - ICU, Emergency, Surgery, Maternity, Laboratory, Imaging, Pediatrics, Pharmacy
- Calculates a **risk level** per region:
  - 🔴 **Critical Desert** — ≤2 services OR ≤3 facilities
  - 🟠 **High Risk** — ≤4 services OR ≤8 facilities
  - 🟡 **Moderate** — ≤6 services OR ≤20 facilities
  - 🟢 **Adequate** — everything else
- **Anomaly Detection** — flags suspicious data patterns:
  - Clinic claiming ICU capability
  - Pharmacy claiming surgery
  - Hospital with zero recorded capabilities
  - Surgery claims without confirmed operating theatre
- Generates interactive **Folium map** of Ghana with risk-coded bubbles
- Logs metrics to **MLflow** experiment tracker
- Saves to Delta tables: `region_gap_analysis`, `facility_anomalies`

### `04_langgraph_rag.py` — LangGraph Pipeline & FAISS Vector Store
- Implements a **3-node LangGraph pipeline** for processing individual hospitals:
  1. `idp_extractor` → extracts facts from description
  2. `anomaly_detector` → checks claims against facility type + gives trust score
  3. `rag_planner` → answers NGO questions using extracted facts
- Embeds all hospitals using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- Builds **FAISS IndexFlatL2** vector index over all hospitals
- Implements `rag_search_v2`:
  - Detects region from question text
  - Pre-filters to that region's hospitals
  - Builds a mini FAISS sub-index for fast regional search
  - Falls back to full search if region is too small
- Adds confidence labels: `High / Medium / Low` per hospital based on data completeness
- Exports CSV, embeddings (`.npy`), and FAISS index (`.faiss`) for Streamlit deployment

### `05_master_test.py` — Quality Validation & Master Testing
- End-to-end validation of all Delta tables
- De-duplicates hospitals (keeps the row with most complete data per name)
- Extended city-to-region mapping fixing 100+ `Unknown` region assignments
- Performance report: extraction rates, region coverage, anomaly counts
- Final data export to `hospital_metadata.csv` for the Streamlit app

### `06_emergency_routing_FINAL.py` — Patient Routing & Doctor Deployment
- Computes Haversine distances between patient regions and hospitals
- Matches medical conditions to required capabilities (e.g., "severe trauma" → surgery, ICU)
- Recommends nearest capable hospitals with travel times
- **Doctor Deployment Optimizer**: Recommends moving doctors from secure/surplus regions to adjacent critical deserts

### `07_fixes_and_evaluation.py` & `08_region_fixes_and_citations.py` — Reranking & Citations
- Extracts JSON arrays back to flat search texts for improved FAISS retrieval
- Maps hard-to-find cities cleanly to their region, solving 100+ "Unknown" regions
- Adds a **Cross-Encoder reranker** (`ms-marco-MiniLM-L-6-v2`) to re-rank vector search results
- Mines descriptions to derive exact sentence-level **citations** for every extracted fact

### `09_pydantic_evaluation_and_extraction.py` — Pydantic Schema Validation
- Formally validates all extracted capabilities against the Virtue Foundation `FacilityFacts` schema using Pydantic
- Validates that `specialties` map strictly to the allowed ontology (e.g. `pediatrics`, `generalSurgery`)

### `10_rag_evaluation_mlflow.py` — RAG 10-Question Evaluation & MLflow
- Evaluates the RAG system using a benchmark dataset of 10 complex NGO questions
- Scores results deterministically and via LLM-as-a-judge (Groq) on 4 metrics: Correctness, Retrieval Quality, Coverage, and Failure Analysis
- Logs all artifacts, scores, and run configurations strictly to **MLflow** for visibility and tracking

### 👨‍⚖️ For Judges: Reproducing Databricks Pipeline
We ran the expensive AI extractions offline to provide a fast user experience. However, to evaluate or reproduce the backend pipeline:
1. Setup a Databricks Cluster (Databricks Runtime 13.3 LTS ML or higher recommended).
2. Install dependencies: `%pip install -U groq folium pydantic rank_bm25 sentence-transformers faiss-cpu`
3. Set your Groq API Key in Databricks secrets or standard environment variable `GROQ_KEY`.
4. Upload all python scripts starting with `01_` to `10_` into your workspace.
5. Run them sequentially (01 → 10). The intermediate steps use Delta Lake for storage.
6. The final output generates the `hospital_metadata.csv` file we bundled in this repository.

---

## ✨ Streamlit Application Features

### 📊 Tab 1 — Dashboard
- **5 KPI metric cards**: Total Facilities, Critical Deserts, Services Tracked, Regions Covered, System Status
- **Facilities by Region bar chart** — color-coded by risk level (Plotly)
- **Risk Distribution donut chart** — breakdown of region risk categories
- **Service Coverage panel** — which services exist across how many regions

### 🔍 Tab 2 — Intelligent Search
- **4 example query buttons** for quick exploration
- **Natural language query box** with intent detection
- **Smart routing**:
  - *"How many hospitals have ICU in Accra?"* → counted across all 896 rows, not just top results
  - *"Which are suspicious facilities?"* → pre-computed anomaly scan
  - *"Where are medical deserts?"* → full regional gap analysis
  - *"Hospitals with maternity in Volta"* → hybrid BM25+FAISS + Groq AI answer
- **AI Analysis panel** — Groq-generated insights with source citations
- **Hospital result cards** — capability, procedures, specialties, confidence badges, NGO tags, anomaly flags
- **Region + service detection context** shown above results

### 📋 Tab 3 — Regional Analysis
- **4 risk-level summary metrics**
- **Interactive gap table** — all regions × 8 services (✓/✗ per cell), filterable by sidebar
- **Deployment Recommendations** — auto-lists critical regions with missing services

### 🗺️ Tab 4 — Interactive Map *(Filter-Aware)*
- **Live Folium map** generated dynamically — responds to sidebar region and risk filters
- Dark/light tile styles (CartoDB dark_matter / positron) matching app theme
- **Circle markers** sized by facility count, coloured by risk
- **Popup detail** per region: risk level, facility count, NGO count, services score, missing services
- **Auto-zoom** to selected region when filter is active

### 🏥 Tab 5 — Hospital Directory
- Full searchable/filterable table of all 896 facilities
- Columns: Hospital, Region, City, Type, Capabilities, Procedures, Specialties, Confidence, NGO flag
- Inline name/city text filter

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.43 | Web application framework |
| **LLM** | Groq LLaMA 3.3 70B Versatile | AI answers, IDP extraction |
| **Vector Search** | FAISS (IndexFlatL2) | Semantic similarity search |
| **Keyword Search** | BM25 (rank_bm25) | Lexical/keyword matching |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | 384-dim text embeddings |
| **Orchestration** | LangGraph | Multi-node AI pipeline |
| **Data** | Pandas + NumPy | Data processing |
| **Mapping** | Folium | Interactive choropleth maps |
| **Charts** | Plotly | Dashboard visualisations |
| **Data Processing** | Databricks + Delta Lake | Cloud ETL pipeline |
| **Experiment Tracking** | MLflow | Model metrics logging |
| **AI Provider** | Groq (free tier) | Fast LLM inference |

---

## 📁 Project Structure

```
medical-desert-agent/
│
├── streamlit_app.py          # Main Streamlit application (1,700 lines)
│                             # Contains: data loading, hybrid search engine,
│                             # question router, LLM integration, all 5 UI tabs
│
├── data/
│   ├── hospital_metadata.csv # 896 Ghana healthcare facilities (enriched)
│   ├── hospital_index.faiss  # Pre-built FAISS vector index
│   ├── hospital_embeddings.npy # Sentence embeddings (all-MiniLM-L6-v2)
│   ├── search_texts.json     # BM25 corpus (rich text per facility)
│   └── ghana_map.html        # Pre-generated Folium map (fallback)
│
├── 01_explore_and_clean.py   # Databricks: data quality + region normalisation
├── 02_idp_agent.py           # Databricks: LLM-powered fact extraction
├── 03_gap_analysis.py        # Databricks: gap scoring + anomaly detection + map
├── 04_langgraph_rag.py       # Databricks: LangGraph pipeline + FAISS indexing
├── 05_master_test.py         # Databricks: end-to-end validation + export
│
├── .streamlit/
│   └── config.toml           # Streamlit theme (teal accent, light base)
│
├── requirements.txt          # Pinned Python dependencies
├── .env                      # GROQ_KEY (not committed to git)
├── .env.example              # Template for contributors
└── .gitignore                # Excludes .env, venv, __pycache__
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (takes 2 minutes to get)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/medical-desert-agent.git
cd medical-desert-agent

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq key:
GROQ_KEY=gsk_your_actual_key_here
```

> **Streamlit Cloud deployment?** Add `GROQ_KEY` to your app's Secrets instead.

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` — the app initialises the search engine on first run (~30 seconds), then all queries are instant.

### 4. Deploying to Streamlit Cloud

To push this application to Streamlit Community Cloud:
1. Fork or push this repository to your GitHub account.
2. Sign in to [Streamlit Share](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and main file path (`streamlit_app.py`).
4. Click **Advanced settings...** and add your Groq secret:
   ```toml
   GROQ_KEY = "gsk_your_actual_key_here"
   ```
5. Click **Deploy!** The platform will automatically install dependencies from `requirements.txt` and launch the app.

---

## 🧠 How the AI Search Works

### Step 1 — Question Routing
Every query is routed before touching the vector store:

```python
"How many hospitals have ICU?"     → COUNT   → pre-computed stats (all 896 rows)
"Any suspicious facilities?"       → ANOMALY → pre-computed anomaly flags
"Which regions need NGOs most?"    → GAP     → regional gap analysis table
"Maternity care in Volta region"   → SEARCH  → hybrid BM25 + FAISS
```

### Step 2 — Query Expansion
Search queries are expanded with medical synonyms:
- `"icu"` → `"icu intensive care unit critical care level-3"`
- `"emergency"` → `"emergency A&E accident trauma 24-hour resuscitation"`

### Step 3 — Hybrid Scoring
```
Final Score = 0.4 × BM25_score + 0.6 × FAISS_score
```
- Region boost: ×1.8 for facilities matching the detected region
- NGO boost: ×2.5 when query mentions NGO/foundation/charity

### Step 4 — LLM Context
The LLM receives **both**:
- Pre-computed stats (accurate counts from all 896 rows)
- Top-5 retrieved examples (specific hospital names to cite)

This prevents the classic RAG hallucination where the LLM says "5 hospitals" because it only saw 5 results.

---

## 📊 Data Insights

| Metric | Value |
|--------|-------|
| Total facilities | 896 unique |
| NGOs identified | ~45+ |
| Regions covered | 16 |
| Critical desert regions | 3–5 (varies by data completeness) |
| Facilities with ICU | ~15-20 |
| Facilities with emergency care | ~120+ |
| Anomalies detected | ~100+ data quality flags |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Search index | FAISS IndexFlatL2 |

---

## 🏆 Hackathon Context

This project was built for the **Virtue Foundation Ghana** hackathon challenge, which asked teams to:

- ✅ Build an IDP (Intelligent Document Processing) agent to extract facility capabilities
- ✅ Implement a synthesis layer to identify regional healthcare gaps
- ✅ Create a planning/routing system for NGO resource deployment
- ✅ (Stretch) Add citation support — every AI answer cites source hospitals
- ✅ (Stretch) Build a map visualisation of medical deserts
- ✅ (Stretch) Real-world impact — actionable recommendations for actual NGO field teams

### Key Technical Differentiators
1. **Question Router bypasses FAISS for count/anomaly queries** — eliminates the #1 RAG failure mode
2. **Pre-computed stats at load time** — Groq always sees accurate counts, not just top-5 examples
3. **Dual search index** — BM25 catches exact keyword matches that FAISS misses; FAISS catches semantic similarity that BM25 misses
4. **Dynamic map** — updates live when region/risk filters change in the sidebar
5. **Trust scoring** — each facility has a confidence level (`High/Medium/Low`) based on data completeness

---

## 🤝 Contributing

Pull requests welcome! Key areas that could use improvement:
- Adding more Ghana regions to the city-to-region mapping
- Improving the IDP extraction prompt for specific facility types (dental, pharmacy)
- Adding real-time map data updates via the Ghana Health Service API

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the people of Ghana · Virtue Foundation Hackathon 2025*
