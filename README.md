# 🩺 HealthGen AI - Intelligent Health RAG Assistant

## 📌 Overview
**HealthGen AI** is a Retrieval-Augmented Generation (RAG) AI assistant designed to answer health-related questions using trusted PDF documents (WHO, CDC, NIH, etc.).  
It provides insights on conditions such as obesity, PCOD/PCOS, asthma, diabetes, hypertension, and general health recommendations.

> ⚠ **Disclaimer:** For informational purposes only. Not a substitute for professional medical advice.

---

## 🧠 Tech Stack
- **Python**  
- **Streamlit** – Interactive Web Interface  
- **PyPDF2** – PDF text extraction  
- **Sentence Transformers** – Text embeddings  
- **FAISS** – Vector search for knowledge retrieval  
- **HuggingFace Transformers** – Flan-T5 language model  

---

## ⚙ Features
1. Upload health-related PDF files.  
2. Extract and split text into overlapping chunks.  
3. Generate embeddings with Sentence Transformers.  
4. Store embeddings in a FAISS vector database.  
5. Retrieve relevant chunks based on user queries.  
6. Generate grounded answers using the language model.  

---

👩‍💻 Author

Jeedipalli Rishma
B.Tech - Data Science
Aspiring AI/ML Engineer

--- 
🤖 Run

pip install -r requirements.txt

streamlit run app.py
