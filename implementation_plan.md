# PropertyLens - House Price Prediction and Support Platform

## Implementation Plan (v2)

A Streamlit web application that estimates residential property prices across **Bangalore, Mumbai and Pune**, explains each estimate, and includes two conversational assistants: a **Price Assistant** for natural-language price queries and a **Help & Support** assistant that answers questions from a knowledge base and escalates to a support ticket when it is unsure.

---

## Product Constraints

- **Zero emojis.** Icons come from Streamlit's Material Symbols set (for example `:material/home:`), which are font glyphs, not Unicode emojis. The `₹` sign is a currency symbol and is allowed.
- **Light theme only.** It is enforced in `.streamlit/config.toml` (`base = "light"`) and backed by a small CSS layer.
- **Currency.** All prices are shown in ₹ Lakhs, and in ₹ Crores at 100 Lakhs or more.
- **Honest data labelling.** The bundled dataset is synthetic but calibrated to approximate 2024-25 locality rates. The Home and Data pages say so clearly. A real dataset can replace it if it uses the same schema (see "Swapping in real data").
- **Works offline.** The app runs fully without an API key. When `ANTHROPIC_API_KEY` is set, both assistants switch to Claude for language understanding and use the rule-based engines as a fallback.

---

## Architecture

```
E:\Harsh\House_Pricing\
├── .streamlit/config.toml        # Light theme
├── app.py                        # Entry point: st.navigation + st.Page routing
├── pyproject.toml                # Dependencies (managed with uv)
├── core/                         # Pure Python, no Streamlit imports - reusable from an API later
│   ├── config.py                 # Paths, city -> locality registry (rates, coordinates)
│   ├── schema.py                 # Pydantic PropertyInput shared by form, chatbot, LLM
│   ├── data_gen.py               # Reproducible synthetic dataset generator (3 cities)
│   ├── features.py               # Model feature frame construction
│   ├── train.py                  # Benchmark + production model training, metrics export
│   ├── predict.py                # Point estimate, price range, contributions, comparables
│   ├── parser.py                 # Rule-based NL extractor with fuzzy locality matching
│   ├── llm.py                    # Optional Claude integration (extraction + grounded support answers)
│   ├── support_kb.py             # Knowledge base loader + TF-IDF retrieval
│   ├── tickets.py                # SQLite support tickets and answer feedback
│   └── finance.py                # EMI, affordability, stamp duty estimates
├── knowledge_base/*.md           # Support FAQ content (one topic per file, sections by "##")
├── ui/components.py              # CSS, cards, formatting helpers
├── views/                        # One file per page (UI only)
│   ├── home.py  predict.py  assistant.py  compare.py
│   ├── explore.py  evaluation.py  support.py
├── data/housing_data.csv         # Generated dataset (6,000 rows)
├── models/                       # Trained artefacts + metrics.json
└── tests/                        # pytest suite
```

---

## 1. Data

**Hierarchy:** City, then Locality. There are 10 localities per city, and each has a base rate per sq.ft, coordinates and a typical metro distance.

| City | Localities (examples) | Base rate range (₹/sq.ft) |
|---|---|---|
| Bangalore | Indiranagar, Koramangala, Jayanagar, HSR Layout, Hebbal, Bellandur, Whitefield, Marathahalli, Yelahanka, Electronic City | 6,000 - 18,500 |
| Mumbai | Worli, Bandra West, Juhu, Powai, Andheri West, Chembur, Goregaon, Borivali, Thane West, Kharghar | 10,500 - 58,000 |
| Pune | Koregaon Park, Kalyani Nagar, Aundh, Viman Nagar, Baner, Kharadi, Wakad, Hinjewadi, Hadapsar, Wagholi | 6,000 - 15,500 |

**Features:** `City`, `Locality`, `Property_Type` (Apartment / Villa / Independent House), `Area_sqft`, `BHK`, `Bathrooms`, `Floor`, `Total_Floors`, `Parking`, `Property_Age`, `Furnishing`, `Has_Gym`, `Has_Pool`, `Has_Lift`, `Metro_Distance_km`.
**Target:** `Price_Lakhs`.

The generator uses non-linear effects (age depreciation curve, metro-distance decay, high-floor premium that is stronger in Mumbai, property-type premium that varies by city) and multiplicative log-normal noise, so models must learn real interactions rather than a single linear formula.

**Swapping in real data:** place a CSV with the same columns at `data/housing_data.csv` and re-run training. Suitable public sources are the Kaggle datasets "Bengaluru House Price Data" and "Housing Prices in Metropolitan Areas of India" (these need column mapping).

---

## 2. Modelling

- **Target transform:** the model learns `log(Price_Lakhs)` and converts predictions back. This handles the 5-10x price gap between Mumbai and Pune.
- **Benchmarks** use 5-fold cross-validation on the training split, then the held-out 20% test split: Linear Regression, Ridge, Random Forest, Gradient Boosting (sklearn), and **LightGBM**.
- **Metrics:** MAE, RMSE, R², **MAPE**. MAPE is the headline number for users ("typically within X%").
- **Production model:** LightGBM with native categorical handling. It is used because it supports:
  - **Price range** through quantile models (P10 / P90), shown as "₹78 - 92 L, most likely ₹85 L".
  - **Per-prediction explanations** through LightGBM's built-in SHAP contributions (`pred_contrib=True`), aggregated per input feature and shown in ₹.
- **Comparable properties:** the 5 nearest listings in the same city, ranked by locality match plus scaled distance on area, BHK and age.
- **Sanity checks** (tests): price rises with area; Worli costs more than Wagholi for the same property; a new property costs more than a 25-year-old one.

Artefacts: `models/lgbm_point.txt`, `models/lgbm_p10.txt`, `models/lgbm_p90.txt`, `models/metrics.json`, `models/test_predictions.csv`.

---

## 3. Price Assistant (chatbot)

- **Slot-filling conversation.** Details collected in earlier turns are kept, so a user can say "3 BHK in Baner", then "about 1400 sq ft", then "make it furnished".
- **Rule-based extractor** (`core/parser.py`) handles:
  - numbers such as "1,200", "1.2k", "around 1200 sq ft", "2.5 bhk"
  - fuzzy locality matching with `rapidfuzz`, including aliases such as "Bandra", "E-City" and "KP"
  - city inference from the locality
  - property type, furnishing, parking, age ("brand new", "ready to move", "10 years old"), floor, amenities
- **Missing information:** the assistant asks for missing *required* fields (location and area or BHK). If only BHK is given, it suggests a typical area and says so. It never silently invents values.
- **Claude mode (optional):** when an API key is present, Claude extracts the fields as structured JSON (validated against the Pydantic schema). The rule-based parser is used if Claude is unavailable or declines.
- **Answer card:** price range, most likely price, price per sq.ft, top three drivers, and a "View full analysis" link to the Predict page with values pre-filled.

## 4. Help & Support (chatbot)

- **Knowledge base:** Markdown files covering:
  - how estimates work and how accurate they are
  - data sources
  - home loans and EMI
  - stamp duty and registration (Karnataka, Maharashtra)
  - RERA basics
  - the buying checklist
  - account and privacy
  - contacting support
- **Retrieval:** TF-IDF with cosine similarity over "##" sections. Answers cite the section title.
  - Rule mode returns the best section directly.
  - Claude mode writes an answer grounded *only* in the retrieved sections.
- **Escalation:** below a confidence threshold, or when the user asks for a human, the assistant offers a ticket form (name, email, category, message). The ticket is stored in SQLite (`data/support.db`) and the user gets a ticket ID such as `PL-000123`.
- **Feedback:** a Helpful / Not helpful button on every answer, logged to SQLite.
- **Routing:** price-style questions asked in Support are pointed to the Price Assistant, and the reverse.

---

## 5. Pages (`st.navigation`)

| Page | Contents |
|---|---|
| Home | Overview, dataset and model stats, how it works, data disclaimer |
| Predict | Form; estimate range; contribution chart; comparable listings; what-if area curve; EMI and affordability calculator; save to comparison list with CSV export |
| Price Assistant | Slot-filling chatbot with quick prompts |
| Compare Localities | City selector; median ₹/sq.ft by locality; map; side-by-side cross-city comparison |
| Explore Data | Filters, table, data dictionary, distributions, price vs area, correlation heatmap, download |
| Model Evaluation | Benchmark table (CV + test), actual vs predicted, residuals, error by city, feature importance |
| Help & Support | Support chatbot, ticket form, ticket lookup by ID |

---

## 6. Tech Stack

| Concern | Choice |
|---|---|
| Environment | `uv` (`uv venv`, `uv pip install -r requirements.txt`) |
| UI | Streamlit 1.4x with `st.navigation`, `st.cache_resource` / `st.cache_data` |
| ML | scikit-learn, LightGBM |
| Charts | Plotly (light template, `scatter_map` for maps) |
| NLP | Regex + `rapidfuzz`; TF-IDF from scikit-learn |
| Validation | Pydantic v2 |
| LLM (optional) | Anthropic Python SDK, `claude-opus-5-5` by default (configurable via `CLAUDE_MODEL`), server-side refusal fallback enabled |
| Storage | SQLite (tickets, feedback) |
| Quality | pytest, ruff |

Secrets go in the `ANTHROPIC_API_KEY` environment variable or in `.streamlit/secrets.toml`, which is git-ignored.

---

## 7. Verification

1. `python -m core.data_gen` writes 6,000 rows, 3 cities and 30 localities.
2. `python -m core.train` writes model artefacts and `metrics.json`, and LightGBM test MAPE is reported.
3. `pytest` covers the parser on 20+ phrasings, model sanity checks, retrieval relevance, the ticket round trip and the EMI maths.
4. Streamlit `AppTest` loads every page without exceptions.
5. An emoji scan of the source tree returns zero matches.
6. `streamlit run app.py` serves HTTP 200.

---

## 8. Later Phases (not in v1)

- Replace synthetic data with scraped or licensed listings, and add a price-trend time series.
- FastAPI endpoint reusing `core/`.
- Rent vs buy calculator.
- PDF export of the analysis.
- Docker image and deployment to Streamlit Community Cloud.
