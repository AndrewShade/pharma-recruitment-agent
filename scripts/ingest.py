import json
import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configuration
JSON_PATH = os.path.join("assets", "ctg-studies.json")
DB_DIR = "./clinical_trial_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def clean_criteria_text(text):
    """Splits eligibility criteria into individual bullet points."""
    text = re.sub(r'(Inclusion|Exclusion) Criteria:', '', text, flags=re.IGNORECASE)
    lines = re.split(r'\n\s*[-*•]\s*|\n\s*\d+\.\s*', text)
    cleaned = [line.strip() for line in lines if len(line.strip()) > 10]
    return cleaned

def run_ingestion():
    print(f"Starting ingestion from {JSON_PATH}...")
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    print(f"Processing {len(data)} clinical trials...")

    for trial in data:
        nct_id = trial['protocolSection']['identificationModule']['nctId']
        criteria_text = trial['protocolSection']['eligibilityModule'].get('eligibilityCriteria', "")
        rules = clean_criteria_text(criteria_text)
        
        for rule in rules:
            doc = Document(
                page_content=rule,
                metadata={"nct_id": nct_id}
            )
            documents.append(doc)

    print(f"Created {len(documents)} searchable rule-chunks.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating ChromaDB index at {DB_DIR}...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print("Ingestion complete. Vector database is ready.")

if __name__ == "__main__":
    run_ingestion()