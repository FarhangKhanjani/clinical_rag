# Clinical Guidelines Q&A

This project builds a question-answering system on top of real clinical guidelines. You ask a plain English question, and it finds the relevant part of the document and gives you a sourced answer.

For example:
- "What is the first line treatment for type 2 diabetes?"
- "What HbA1c level is used for diagnosis?"
- "What lifestyle changes are recommended for diabetes prevention?"

---

## How it works

The system uses a technique called RAG — Retrieval-Augmented Generation.

Instead of feeding a 112-page PDF to an AI model every time, it first searches for the most relevant paragraphs, then passes only those to the model to generate an answer. This makes it faster, cheaper, and more accurate than just uploading the document to a chatbot.

The pipeline has three main parts:

**1. Indexing (runs once)**
The PDF gets split into small chunks of around 500 characters. Each chunk is converted into a list of numbers that represents its meaning — this is called an embedding. All the embeddings are stored in a local vector database (ChromaDB).

**2. Retrieval (runs on each question)**
The question gets converted into an embedding using the same method. The database finds the 5 chunks whose embeddings are closest to the question — closest meaning most similar in meaning.

**3. Generation (runs on each question)**
The 5 retrieved chunks get passed to Claude as context. Claude reads them and writes a clean, sourced answer. The prompt tells Claude to only use the provided context, which prevents it from making things up.

---

## Document used

IDF Global Clinical Practice Recommendations for Managing Type 2 Diabetes — 2025

Published by the International Diabetes Federation. Freely available at idf.org.

---

## Results

The system correctly answers clinical questions by retrieving relevant sections and generating grounded responses. It cites the section of the document the answer came from.

Example output for "What are the first line medications for treating type 2 diabetes?":

> Based on the guidelines, metformin is recommended as the first line pharmacological treatment for type 2 diabetes in most patients, unless contraindicated. This comes from the Pharmacological Management section of the IDF Global Clinical Practice Recommendations for Managing Type 2 Diabetes — 2025.

---

## Stack

```
Python, pypdf, LangChain, ChromaDB, sentence-transformers, Anthropic API
```

---

## How to run

```bash
git clone https://github.com/farhangkhanjani/clinical-rag
cd clinical-rag
pip install -r requirements.txt
```

Download the IDF guidelines PDF from idf.org and place it at `data/diabetes_guidelines.pdf`.

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-key-here
```

Then open and run `notebooks/rag_pipeline.ipynb`.

The first run will download the embedding model (~80MB) and build the vector database. After that, only the retrieval and generation steps run.

---

## Project structure

```
clinical-rag/
├── README.md
├── requirements.txt
├── .env               ← your API key (never pushed to GitHub)
├── .gitignore
├── data/
│   └── chroma_db/     ← vector database (auto-generated, not pushed)
└── notebooks/
    └── rag_pipeline.ipynb
```

---

## What comes next

This is Project 3 in a series focused on clinical data science. The next step is to add a simple interface so non-technical users can type questions and get answers without opening a notebook.
