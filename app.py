import streamlit as st
import json
import os
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

class TrialEvaluation(BaseModel):
    """Schema for validated clinical trial eligibility assessment."""
    nct_id: str = Field(description="National Clinical Trial identifier")
    verdict: Literal["MATCH", "MISMATCH", "POTENTIAL"] = Field(description="Eligibility determination")
    match_confidence: int = Field(description="Subjective confidence score (1-10)")
    reasoning: List[str] = Field(description="Clinical logic points supporting the verdict")
    data_gaps: List[str] = Field(description="Identified missing clinical parameters")
    follow_up_questions: List[str] = Field(description="Targeted questions for clinical verification")

st.set_page_config(
    page_title="Clinical Trial Matcher", 
    page_icon="🧪", 
    layout="wide"
)

@st.cache_resource
def initialize_backend():
    """Initializes local vector store and multi-stage inference models."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(
        persist_directory="./clinical_trial_index", 
        embedding_function=embeddings
    )
    
    # Primary reasoning model for clinical logic
    reasoning_llm = OllamaLLM(model="mistral-nemo", temperature=0, extra_kwargs={"seed": 42})
    
    # Secondary model optimized for structured extraction
    extraction_llm = ChatOllama(model="mistral-nemo", temperature=0).with_structured_output(TrialEvaluation)
    
    with open(r"assets/ctg-studies.json", "r", encoding="utf-8") as f:
        json_obj = json.load(f)
        
    trial_lookup = {
        item['protocolSection']['identificationModule']['nctId']: 
        item['protocolSection']['eligibilityModule']['eligibilityCriteria']
        for item in json_obj
    }
    
    return vector_db, reasoning_llm, extraction_llm, trial_lookup

# App UI
st.title("🧪 Agentic Clinical Trial Matching")
st.markdown("Context-aware screening using a multi-stage reasoning and extraction pipeline.")

try:
    vector_db, reasoning_llm, extraction_llm, trial_lookup = initialize_backend()
except Exception as e:
    st.error(f"Initialization failed. Check local Ollama service. Error: {e}")
    st.stop()

with st.sidebar:
    st.header("Patient Input")
    uploaded_file = st.file_uploader("Upload Patient Note", type=['json', 'txt'])
    st.divider()
    st.info("System utilizing local 24GB VRAM for secure, on-device inference.")

if uploaded_file:
    if uploaded_file.name.endswith('.json'):
        raw_json = json.load(uploaded_file)
        patient_summary = raw_json.get('clinical_summary', raw_json)
        patient_text = json.dumps(patient_summary, indent=2)
    else:
        patient_text = uploaded_file.read().decode("utf-8")

    st.subheader("Patient Clinical Profile")
    st.code(patient_text, language='json')

    if st.button("Evaluate Eligibility"):
        all_results = []
        
        with st.spinner("Executing multi-stage inference pipeline..."):
            # Semantic search for relevant protocol sections
            search_results = vector_db.similarity_search(patient_text, k=10)
            unique_ids = list(set([res.metadata['nct_id'] for res in search_results]))
            
            # Clinical reasoning template
            thinking_prompt = ChatPromptTemplate.from_template("""
            Analyze the following patient against the clinical trial eligibility criteria. 
            Prioritize timeline consistency and primary diagnostic markers.

            PATIENT: {patient_note}
            TRIAL CRITERIA: {trial_rule}

            OUTPUT:
            REASONING: Detailed bullet points for clinical alignment.
            VERDICT: MATCH, MISMATCH, or POTENTIAL.
            FOLLOW-UP: Specific missing information required for verification.
            """)

            # Structuring template for Pydantic validation
            extraction_prompt = ChatPromptTemplate.from_template("""
            Map the following clinical analysis to the validated schema. 
            Strictly preserve the auditor's original logic and verdict.

            ANALYSIS: {analysis}
            """)

            for nct_id in unique_ids:
                full_criteria = trial_lookup.get(nct_id, "Criteria not found.")
                
                # Phase 1: Clinical Reasoning
                raw_analysis = (thinking_prompt | reasoning_llm).invoke({
                    "patient_note": patient_text,
                    "trial_rule": full_criteria
                })
                
                # Phase 2: Structured Output Generation
                try:
                    structured_output = (extraction_prompt | extraction_llm).invoke({"analysis": raw_analysis})
                    structured_output.nct_id = nct_id
                    all_results.append(structured_output)
                except:
                    continue 

        st.divider()
        st.header("Eligibility Assessment Report")

        tab1, tab2, tab3 = st.tabs(["✅ Matches", "⚠️ Potential", "❌ Mismatches"])

        with tab1:
            matches = [r for r in all_results if r.verdict == "MATCH"]
            if not matches: st.write("No definitive matches identified.")
            for m in matches:
                with st.expander(f"Trial {m.nct_id} (Confidence: {m.match_confidence}/10)"):
                    st.write("**Assessment Reasoning:**")
                    for point in m.reasoning:
                        st.write(f"- {point}")
                    if m.data_gaps:
                        st.caption(f"Verification Required: {', '.join(m.data_gaps)}")

        with tab2:
            potentials = [r for r in all_results if r.verdict == "POTENTIAL"]
            if not potentials: st.write("No potential matches identified.")
            for p in potentials:
                with st.expander(f"Trial {p.nct_id}"):
                    st.warning("Action Required: Missing Critical Parameters")
                    for q in p.follow_up_questions:
                        st.markdown(f"**Clinical Query:** {q}")
                    st.write("**Reasoning:**")
                    for point in p.reasoning:
                        st.write(f"- {point}")

        with tab3:
            mismatches = [r for r in all_results if r.verdict == "MISMATCH"]
            if not mismatches: st.write("No protocol violations identified.")
            for nm in mismatches:
                with st.expander(f"Trial {nm.nct_id}"):
                    st.error("Exclusion Criteria Met / Protocol Violation")
                    for point in nm.reasoning:
                        st.write(f"- {point}")

else:
    st.info("Awaiting clinical note upload for screening.")