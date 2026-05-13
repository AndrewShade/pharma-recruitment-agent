import streamlit as st
import json
import os
import logging
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.engine import ClinicalEngine

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_BASE_DIR, "clinical_trial_index")
_ASSETS_PATH = os.path.join(_BASE_DIR, "assets", "ctg-studies.json")

def _check_ollama() -> bool:
    """Returns True if the local Ollama service is reachable."""
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

st.set_page_config(
    page_title="Clinical Trial Matcher", 
    layout="wide"
)

@st.cache_resource
def initialize_system() -> tuple:
    """Loads embeddings, vector DB, inference engine, and trial lookup. Cached for the session."""
    # Force embeddings to CPU to prevent VRAM conflicts with Ollama
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    if not os.path.exists(_DB_DIR):
        raise RuntimeError("Vector database not found. Run 'python scripts/ingest.py' first.")

    vector_db = Chroma(
        persist_directory=_DB_DIR,
        embedding_function=embeddings
    )

    engine = ClinicalEngine()

    with open(_ASSETS_PATH, "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    
    trial_lookup = {
        item['protocolSection']['identificationModule']['nctId']: 
        item['protocolSection']['eligibilityModule']['eligibilityCriteria']
        for item in json_obj
    }
    
    return vector_db, engine, trial_lookup

@st.cache_data(show_spinner=False)
def _cached_evaluate(_engine, nct_id: str, patient_text: str, trial_rule: str):
    """Evaluates one trial against one patient. Cached per (nct_id, patient_text) pair."""
    return _engine.evaluate(nct_id, patient_text, trial_rule)

# Application Logic
st.title("Clinical Trial Matching System")
st.markdown("Automated patient screening via two-stage clinical reasoning and data extraction.")

if not _check_ollama():
    st.error("Ollama is not running. Please start Ollama before using this app.")
    st.stop()

try:
    vector_db, engine, trial_lookup = initialize_system()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

with st.sidebar:
    st.header("Patient Input")
    uploaded_file = st.file_uploader("Upload Clinical Note", type=['json', 'txt'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.json'):
            raw_json = json.load(uploaded_file)
            patient_summary = raw_json.get('clinical_summary', raw_json)
            patient_text = json.dumps(patient_summary, indent=2)
        else:
            patient_text = uploaded_file.read().decode("utf-8")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.subheader("Extracted Profile")
    st.code(patient_text, language='json')

    if st.button("Run Evaluation"):
        with st.status("Analyzing protocols...", expanded=True) as status:
            search_results = vector_db.similarity_search(patient_text, k=10)
            unique_ids = list(set([res.metadata['nct_id'] for res in search_results]))
            all_results = []
            for i, nct_id in enumerate(unique_ids, 1):
                status.write(f"Evaluating trial {i}/{len(unique_ids)}: {nct_id}")
                full_criteria = trial_lookup.get(nct_id)
                if not full_criteria:
                    continue
                try:
                    evaluation = _cached_evaluate(engine, nct_id, patient_text, full_criteria)
                    all_results.append(evaluation)
                except Exception as e:
                    logging.warning("Trial %s evaluation failed: %s", nct_id, e)
            status.update(
                label=f"Complete — {len(all_results)}/{len(unique_ids)} trials evaluated",
                state="complete",
                expanded=False
            )

        tab1, tab2, tab3 = st.tabs(["Matches", "Potentials", "Mismatches"])

        with tab1:
            matches = [r for r in all_results if r.verdict == "MATCH"]
            if not matches:
                st.write("No high-confidence matches identified.")
            for m in matches:
                with st.expander(f"Trial {m.nct_id} - Confidence: {m.match_confidence}/100"):
                    st.write("**Reasoning:**")
                    for point in m.reasoning:
                        st.write(f"- {point}")
                    if m.data_gaps:
                        st.caption(f"Secondary Verification Needed: {', '.join(m.data_gaps)}")

        with tab2:
            potentials = [r for r in all_results if r.verdict == "POTENTIAL"]
            if not potentials:
                st.write("No potential matches identified.")
            for p in potentials:
                with st.expander(f"Trial {p.nct_id}"):
                    st.warning("Missing Critical Eligibility Data")
                    for q in p.follow_up_questions:
                        st.markdown(f"**Required Verification:** {q}")
                    st.write("**Analysis:**")
                    for point in p.reasoning:
                        st.write(f"- {point}")

        with tab3:
            mismatches = [r for r in all_results if r.verdict == "MISMATCH"]
            if not mismatches:
                st.write("No exclusions found.")
            for nm in mismatches:
                with st.expander(f"Trial {nm.nct_id}"):
                    st.error("Exclusion Criteria Met")
                    for point in nm.reasoning:
                        st.write(f"- {point}")
else:
    st.info("Upload a patient file to begin matching.")