# Cleaning, segmentation helpers
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def clause_segmentation(text: str):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    merged = []
    for s in sentences:
        if len(s.split()) < 4 and merged:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged
