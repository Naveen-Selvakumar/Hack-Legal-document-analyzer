# # Main Streamlit app (UI + backend logic)
# import streamlit as st

# from transformers import pipeline
# import spacy

# from utils.text_extract import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
# from utils.preprocess import preprocess_text, clause_segmentation
# from utils.granite import chunk_text


# # ------------------ Load HuggingFace pipelines ------------------

# import torch
# device = 0 if torch.cuda.is_available() else -1

# @st.cache_resource
# def load_pipelines():
#     # Try to load large models, fallback to lightweight or None on OOM
#     try:
#         zero_shot = pipeline(
#             "zero-shot-classification",
#             model="facebook/bart-large-mnli",
#             device=device
#         )
#     except Exception as e:
#         st.warning("Zero-shot classification model could not be loaded due to memory constraints. Classification will be disabled.")
#         zero_shot = None

#     try:
#         simplifier = pipeline(
#             "text2text-generation",
#             model="google/flan-t5-small",
#             device=device
#         )
#     except Exception as e:
#         st.warning("Text simplification model could not be loaded. Simplification will be disabled.")
#         simplifier = None

#     try:
#         ner = pipeline(
#             "ner",
#             model="dslim/bert-base-NER",
#             aggregation_strategy="simple",
#             device=device
#         )
#     except Exception as e:
#         st.warning("NER model could not be loaded. Entity extraction will be disabled.")
#         ner = None

#     try:
#         nlp = spacy.load("en_core_web_sm")
#     except Exception as e:
#         st.warning("spaCy model could not be loaded. Clause segmentation will be disabled.")
#         nlp = None

#     return {
#         "zero_shot": zero_shot,
#         "simplifier": simplifier,
#         "ner": ner,
#         "nlp": nlp
#     }

# def simplify_clause(summarizer, clause: str, max_length: int = 60) -> str:
#     if len(clause.split()) < 6:
#         return clause
#     try:
#         # For short inputs, set max_length to input length + 2
#         input_len = len(clause.split())
#         max_len = min(max_length, input_len + 2)
#         out = summarizer(clause, max_length=max_len, min_length=8, do_sample=False)
#         return out[0]["summary_text"].strip()
#     except:
#         return clause.split(".")[0]

# def extract_entities(ner_pipeline, text: str):
#     ents = ner_pipeline(text)
#     return [{"entity": e.get("entity_group"), "word": e.get("word"), "score": float(e["score"])} for e in ents]

# def classify_document(zero_shot_pipeline, text: str, labels):
#     out = zero_shot_pipeline(text, candidate_labels=labels)
#     return out

# # ------------------ Streamlit UI ------------------
# st.set_page_config(page_title="ClauseWise — Legal Document Analyzer", layout="wide")
# st.title(" ClauseWise — Legal Document Analyzer")
# st.caption("Upload a PDF, DOCX, or TXT to analyze clauses, entities, and document type.")

# uploaded = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

# if uploaded:
#     raw_bytes = uploaded.read()

#     if uploaded.name.lower().endswith(".pdf"):
#         text = extract_text_from_pdf(raw_bytes)
#     elif uploaded.name.lower().endswith(".docx"):
#         text = extract_text_from_docx(raw_bytes)
#     else:
#         text = extract_text_from_txt(raw_bytes)

#     text = preprocess_text(text)

#     st.header("Document Preview")
#     st.text_area("Document Content", text[:1000] + ("..." if len(text) > 1000 else ""), height=200)

#     with st.spinner("Loading AI pipelines..."):
#         pipes = load_pipelines()


#     # Direct function calls
#     # For large docs, chunk and process sequentially
#     # Clause segmentation
#     clauses = clause_segmentation(text)

#     # Clause simplification
#     simplified = []
#     for c in clauses:
#         clause_chunks = chunk_text(c, max_words=100)
#         simplified_chunks = []
#         for chunk in clause_chunks:
#             if pipes["simplifier"] is not None:
#                 try:
#                     out = pipes["simplifier"](chunk, max_length=60, min_length=8, do_sample=False)
#                     simplified_chunk = out[0].get("generated_text") or out[0].get("summary_text")
#                     simplified_chunks.append(simplified_chunk.strip())
#                 except Exception:
#                     simplified_chunks.append(chunk)
#             else:
#                 simplified_chunks.append(chunk)
#         simplified.append(" ".join(simplified_chunks))

#     # Document summary (optional, using simplifier)
#     summary = None
#     if pipes["simplifier"] is not None:
#         try:
#             out = pipes["simplifier"](text[:1000], max_length=60, min_length=15, do_sample=False)
#             summary = out[0].get("generated_text") or out[0].get("summary_text")
#         except Exception:
#             summary = None

#     # Entity extraction
#     entities = extract_entities(pipes["ner"], text) if pipes["ner"] is not None else []

#     # Document classification
#     doc_type = None
#     if pipes["zero_shot"] is not None:
#         try:
#             doc_type = classify_document(
#                 pipes["zero_shot"],
#                 text[:1000],
#                 ["Employment Agreement", "Lease Agreement", "Sales Agreement",
#                  "NDA", "Service Agreement", "Power of Attorney", "Affidavit", "Will"]
#             )
#         except Exception:
#             doc_type = None

#     # Display Results
#     st.header("AI Analysis Results")

#     # Summary and important notes
#     if summary:
#         st.subheader("Document Summary")
#         st.success(summary)

#     # Document type
#     if doc_type and "labels" in doc_type:
#         st.subheader("Document Type Prediction")
#         for lbl, score in zip(doc_type["labels"], doc_type["scores"]):
#             st.write(f"- {lbl}: {score:.2f}")

#     # Named entities
#     st.subheader("Named Entities")
#     if entities:
#         for e in entities:
#             st.write(f"- {e['entity']}: {e['word']} ({e['score']:.2f})")
#     else:
#         st.write("No entities found.")

#     # Clause breakdown
#     st.subheader("Clause Breakdown & Simplification")
#     for i, (orig, simp) in enumerate(zip(clauses, simplified), 1):
#         with st.expander(f"Clause {i}"):
#             st.markdown(f"**Original:**\n{orig[:500]}{'...' if len(orig)>500 else ''}")
#             st.markdown(f"**Simplified:**\n{simp[:500]}{'...' if len(simp)>500 else ''}")
#         for lbl, score in zip(doc_type["labels"], doc_type["scores"]):
#             st.write(f"- {lbl}: {score:.2f}")

# else:
#     st.info("Please upload a PDF, DOCX, or TXT file.")

import streamlit as st
import spacy
import docx
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------------
# 1. Load Granite & other pipelines
# -------------------------------
# In your main app file (e.g., app.py or main.py)
# ...
@st.cache_resource
def load_pipelines():
    # Ensure this matches the model name you intend to use and that exists
    # If you want to use the smaller 350m variant, ensure that specific model ID is correct on HF
    # For now, let's assume you meant the 1B model from utils/granite.py
    granite_model_name = "ibm-granite/granite-3.1-1b-a400m-instruct" # <--- **CHANGE THIS LINE**
    tokenizer = AutoTokenizer.from_pretrained(granite_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        granite_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    # ... rest of your pipeline loading
    # Create a text-generation pipeline
    simplifier_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=False
    )

    # NER pipeline
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    # Zero-shot classification
    classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    return simplifier_pipeline, ner_pipeline, classifier_pipeline

simplifier_pipeline, ner_pipeline, classifier_pipeline = load_pipelines()

# -------------------------------
# 2. Clause segmentation using SpaCy
# -------------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

def segment_clauses(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

# -------------------------------
# 3. Multi-format file loader
# -------------------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return None

# -------------------------------
# 4. Streamlit App UI
# -------------------------------
st.title("📜 ClauseWise - Legal Document Analyzer (IBM Granite + Hugging Face)")

uploaded_file = st.file_uploader("Upload a legal document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file:
    raw_text = extract_text_from_file(uploaded_file)
    st.subheader("📄 Extracted Text")
    st.write(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

    # Step 1: Clause segmentation
    clauses = segment_clauses(raw_text)
    st.subheader("🔹 Detected Clauses")
    for i, clause in enumerate(clauses[:10]):  # show first 10 clauses
        st.write(f"**Clause {i+1}:** {clause}")

    # Step 2: Clause simplification using Granite
    st.subheader("✨ Simplified Clauses")
    for i, clause in enumerate(clauses[:10]):
        simplified = simplifier_pipeline(clause)[0]["generated_text"]
        st.write(f"**Simplified Clause {i+1}:** {simplified}")

    # Step 3: Named Entity Recognition
    st.subheader("🧾 Named Entities")
    entities = ner_pipeline(raw_text[:1000])  # first 1000 chars for speed
    for ent in entities:
        st.write(f"- {ent['word']} ({ent['entity_group']})")

    # Step 4: Document Type Classification
    st.subheader("📂 Document Type")
    candidate_labels = ["Non-Disclosure Agreement", "Lease Agreement", "Employment Contract", "Service Agreement"]
    classification = classifier_pipeline(raw_text[:1000], candidate_labels)
    st.write(f"Predicted Type: **{classification['labels'][0]}** (score: {classification['scores'][0]:.2f})")

    # Step 5: Optional: Summarization of entire document
    st.subheader("📝 Document Summary")
    summary = simplifier_pipeline(raw_text[:2000])[0]["generated_text"]
    st.write(summary)
