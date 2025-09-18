# Main Streamlit app (UI + backend logic)
import streamlit as st

from transformers.pipelines import pipeline

from utils.text_extract import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from utils.preprocess import preprocess_text, clause_segmentation
from utils.granite import generate_with_granite, chunk_text


# ------------------ Load HuggingFace pipelines ------------------

import torch
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_pipelines():
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small",
        device=device
    )
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=device
    )
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    return {"simplifier": summarizer, "ner": ner, "zero_shot": zero_shot}

def simplify_clause(summarizer, clause: str, max_length: int = 60) -> str:
    if len(clause.split()) < 6:
        return clause
    try:
        # For short inputs, set max_length to input length + 2
        input_len = len(clause.split())
        max_len = min(max_length, input_len + 2)
        out = summarizer(clause, max_length=max_len, min_length=8, do_sample=False)
        return out[0]["summary_text"].strip()
    except:
        return clause.split(".")[0]

def extract_entities(ner_pipeline, text: str):
    ents = ner_pipeline(text)
    return [{"entity": e.get("entity_group"), "word": e.get("word"), "score": float(e["score"])} for e in ents]

def classify_document(zero_shot_pipeline, text: str, labels):
    out = zero_shot_pipeline(text, candidate_labels=labels)
    return out

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="ClauseWise — Legal Document Analyzer", layout="wide")
st.title("📄 ClauseWise — Legal Document Analyzer")
st.caption("Upload a PDF, DOCX, or TXT to analyze clauses, entities, and document type.")

uploaded = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

if uploaded:
    raw_bytes = uploaded.read()

    if uploaded.name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(raw_bytes)
    elif uploaded.name.lower().endswith(".docx"):
        text = extract_text_from_docx(raw_bytes)
    else:
        text = extract_text_from_txt(raw_bytes)

    text = preprocess_text(text)

    st.header("Preview")
    st.text_area("Document Content", text[:1000] + ("..." if len(text) > 1000 else ""), height=200)

    with st.spinner("Loading AI pipelines..."):
        pipes = load_pipelines()


    # Direct function calls
    # For large docs, chunk and process sequentially
    clauses = clause_segmentation(text)
    simplified = []
    for c in clauses:
        # Chunk each clause for memory-efficient generation
        clause_chunks = chunk_text(c, max_words=100)
        simplified_chunks = []
        for chunk in clause_chunks:
            # Use a safe max_new_tokens for CPU/RAM
            simplified_chunk = generate_with_granite(chunk, max_new_tokens=128)
            simplified_chunks.append(simplified_chunk.strip())
        # Join back the simplified chunks for this clause
        simplified.append(" ".join(simplified_chunks))
    entities = extract_entities(pipes["ner"], text)
    doc_type = classify_document(
        pipes["zero_shot"],
        text[:1000],
        ["Employment Agreement", "Lease Agreement", "Sales Agreement",
         "NDA", "Service Agreement", "Power of Attorney", "Affidavit", "Will"]
    )

    # Display Results
    st.header("Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Simplified Clauses")
        for i, (orig, simp) in enumerate(zip(clauses, simplified), 1):
            with st.expander(f"Clause {i}"):
                st.markdown(f"**Original:**\n{orig[:500]}{'...' if len(orig)>500 else ''}")
                st.markdown(f"**Simplified:**\n{simp[:500]}{'...' if len(simp)>500 else ''}")

    with col2:
        st.subheader("Named Entities")
        if entities:
            for e in entities:
                st.write(f"- {e['entity']}: {e['word']} ({e['score']:.2f})")
        else:
            st.write("No entities found.")

        st.subheader("Document Type Prediction")
        for lbl, score in zip(doc_type["labels"], doc_type["scores"]):
            st.write(f"- {lbl}: {score:.2f}")

else:
    st.info("Please upload a PDF, DOCX, or TXT file.")
