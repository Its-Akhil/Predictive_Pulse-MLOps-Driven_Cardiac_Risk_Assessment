# 🫀 Heart Failure Analysis Dashboard

A Streamlit-powered dashboard for exploring heart-failure clinical records and driving interactive mortality-risk predictions via MLflow. Designed for rapid prototyping, reproducibility, and seamless local or cloud deployment.

---

## 🔍 Overview

This repository hosts a two-mode Streamlit app:

1. **Exploratory Data Analysis**  
   - Interactive tables, metrics, and plots (histograms, bar charts, correlation heatmap) for understanding feature distributions and target imbalance.  
2. **Interactive Prediction**  
   - A form-based UI to enter patient features (11 clinical covariates).  
   - On-the-fly risk prediction using an MLflow-logged `pyfunc` model, with a dynamic signature lookup to enforce correct feature ordering.

Under the hood:  
- **Streamlit** for UI & caching (`@st.cache_data`, `@st.cache_resource`) to avoid redundant I/O and model reloads :contentReference[oaicite:0]{index=0}.  
- **Pandas/NumPy** for data wrangling.  
- **Seaborn/Matplotlib** for plotting.  
- **MLflow** for model versioning, packaging, and loading :contentReference[oaicite:1]{index=1}.

---

## 📁 File Structure

```text
.
├── app/
│   ├── models/                         
│   │   └── └── MLflow artifacts (logged model)
│   ├── streamlit_app.py                # Core app with EDA & prediction modes
│   └── mlflow_local_integration.py     # Utility: load MLflow model via URI
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml                    # (optional) MLflow tracking URI & tokens


```
```
mlflow_local_integration.py
Provides a thin wrapper around mlflow.pyfunc.load_model(model_uri) allowing you to point to any MLflow model (local file path, run URI, or registry URI) without modifying the main streamlit_app.py.
```

⚙️ Installation

  1. Clone & enter the repo
    git clone [https://github.com/Its-Akhil/heart-failure-prediction-dashboard.git]
    cd heart-failure-dashboard

  2. Python environment
        [python ]
        python3 -m venv venv
        source venv/bin/activate      # macOS/Linux
        venv\Scripts\activate.bat     # Windows
  
  4. Dependencies
    pip install -r requirements.txt

  5. Data file
    Place heart_failure_clinical_records_dataset.csv under data/.
    If missing, fetch from the UCI ML Repository ( https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records )
    

🚀 Running Locally

  streamlit run app/streamlit_app.py
  By default, the sidebar will try to load the model from app/models/ via mlflow.pyfunc.load_model.

To override, use the CLI flag:
```
  STREAMLIT_SERVER_RUN_ON_SAVE=true 
  MODEL_URI="runs:/<RUN_ID>/model" 
  streamlit run app/streamlit_app.py
```
  This delegates to mlflow_local_integration.py, which calls:
```
  import mlflow.pyfunc
  mlflow.pyfunc.load_model(model_uri=os.getenv("MODEL_URI"))
```
  
  🔧 mlflow_local_integration.py
  ```
  import os
  import mlflow.pyfunc
  
  def load_model_from_uri():
      uri = os.getenv("MODEL_URI", "./app/models")
      return mlflow.pyfunc.load_model(uri)
```
  Flexibility: point MODEL_URI to any MLflow URI (runs:/…, models:/…, or local path).  
  Separation of concerns: keeps model-loading logic decoupled from UI code 


📝 Key Technical Highlights
  1. Streamlit Caching
    @st.cache_data caches the CSV load.
    @st.cache_resource caches the MLflow model object, preventing repeated downloads from remote tracking servers 
  2. Dynamic Signature Handling
    The app queries model.metadata.get_input_schema().input_names() to build input DataFrame in the exact order expected by the model, making it robust to flavor changes.
  3. Imbalanced-Target Visualization
    Pie charts and metrics for the binary DEATH_EVENT flag demonstrate class imbalance upfront, guiding users to consider resampling or class-weighting strategies before retraining.
  4. Extensible Architecture
    Drop-in alternative models: swap mlflow.pyfunc.load_model URIs.
    Advanced users can integrate SHAP or LIME explainer modules in the EDA section by adding new cacheable functions.

  
☁️ Deploying on Streamlit Cloud
  Push your fork to GitHub.
  1. In Streamlit Cloud, “New app” → link your repo → set app/streamlit_app.py as the entrypoint.
  2. Under Advanced settings, add secrets:

    [mlflow]
    tracking_uri = "https://your-mlflow-server"
    token        = "••••••"
    Optionally set MODEL_URI env var to point at a registry model, e.g. models:/heart_dashboard/Production.


🤝 Contributing & Roadmap
Issues & PRs welcome!

Next steps:
  
  Add model-drift monitoring via Evidently or custom MLflow checks.
  Dockerize the entire stack (Streamlit + MLflow tracking server).
  Integrate federated learning or privacy-preserving layers for multi-center data.
  
  
“A model is only as good as its reproducibility—MLflow and Streamlit make it delightfully traceable.”
  – Your unapologetically nerdy AI mentor
