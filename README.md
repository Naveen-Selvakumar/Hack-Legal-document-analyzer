# ClauseWise

ClauseWise is an AI-powered legal document analyzer designed to simplify, decode, and classify complex legal texts for lawyers, businesses, and laypersons alike.

## Project Structure

```
clausewise/
│── app.py                  # Main Streamlit app (UI + backend logic)
│── requirements.txt        # Dependencies (streamlit, transformers, spacy, pdfplumber, python-docx, etc.)
│── models/                 # (Optional) Store custom fine-tuned models
│── utils/
│     ├── text_extract.py   # Functions for PDF/DOCX/TXT extraction
│     ├── preprocess.py     # Cleaning, segmentation helpers
│     └── granite.py        # Orchestration pipeline (Granite helper)
│── data/                   # (Optional) Sample documents to test
│── README.md               # Documentation
```

## Features
- Clause simplification (Granite model)
- Named Entity Recognition (NER)
- Clause segmentation
- Document type classification
- Multi-format document support (PDF, DOCX, TXT)
- Streamlit UI

## Setup
```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run
```sh
streamlit run app.py
```

- fastapi, uvicorn, streamlit, python-docx, PyPDF2, transformers, torch, ibm-watson
- **pdfplumber** (PDF extraction)
- **docx2txt** (DOCX extraction)
- **spacy** (clause segmentation)

## Running the App

### Backend

```sh
cd backend
uvicorn app:app --reload
```

### Frontend

```sh
cd frontend
streamlit run streamlit_app.py
```
