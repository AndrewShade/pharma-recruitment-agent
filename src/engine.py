import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the Structured Output Schema
class TrialEvaluation(BaseModel):
    """The result of evaluating a patient against a clinical trial."""
    nct_id: str = Field(description="The unique identifier for the clinical trial")
    verdict: Literal["MATCH", "MISMATCH", "POTENTIAL"] = Field(description="The final decision")
    match_confidence: int = Field(description="Score from 1 to 10 on certainty")
    reasoning: List[str] = Field(description="Step by step logical confirmation points")
    data_gaps: List[str] = Field(description="Missing secondary info like minor labs")
    follow_up_questions: List[str] = Field(description="Critical showstoppers only")

class ClinicalEngine:
    def __init__(self, model_name="mistral-nemo"):
        # The Worker (Logic)
        self.worker_llm = OllamaLLM(
            model=model_name, 
            temperature=0, 
            extra_kwargs={"seed": 42}
        )
        
        # The Clerk (Structure)
        self.clerk_llm = ChatOllama(
            model=model_name, 
            temperature=0
        ).with_structured_output(TrialEvaluation)

        self.thinking_prompt = ChatPromptTemplate.from_template("""
            You are a Senior Clinical Trial Auditor. Use the following hierarchy of logic:
            1. TIMELINE CHECK: Is the patient currently on a medication that should be "Prior" or "Stopped"?
            2. CONDITION CHECK: Does the patient's Stage/Diagnosis match the rule?
            3. AMBIGUITY CHECK: Is there a core piece of data missing?

            PATIENT: {patient_note}
            TRIAL CRITERIA: {trial_rule}

            OUTPUT FORMAT:
            REASONING: <your step by step logic>
            VERDICT: <MATCH, MISMATCH, or POTENTIAL>
            FOLLOW-UP: <CRITICAL questions only>
        """)

        self.extraction_prompt = ChatPromptTemplate.from_template("""
            Convert this clinical analysis into a structured format. 
            Strictly follow the intent: If a contradiction exists, it is a MISMATCH. 
            If core mutations are unknown, it is POTENTIAL.

            ANALYSIS: {analysis}
        """)

    def evaluate(self, nct_id, patient_text, trial_rule):
        # Stage 1: Raw Reasoning
        reasoning_chain = self.thinking_prompt | self.worker_llm
        raw_analysis = reasoning_chain.invoke({
            "patient_note": patient_text,
            "trial_rule": trial_rule
        })
        
        # Stage 2: Structured Extraction
        extraction_chain = self.extraction_prompt | self.clerk_llm
        structured_output = extraction_chain.invoke({"analysis": raw_analysis})
        
        # Binding metadata
        structured_output.nct_id = nct_id
        return structured_output