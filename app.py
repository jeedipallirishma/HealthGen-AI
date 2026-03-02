import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Streamlit page setup
st.set_page_config(page_title="HealthGen AI - RAG Assistant", layout="wide")
st.title("🩺 HealthGen AI - Intelligent Health RAG Assistant")

st.markdown("""
This AI assistant answers health-related queries such as:
- Obesity management
- PCOD / PCOS awareness
- Asthma precautions
- Diabetes lifestyle guidance
- Hypertension and general health recommendations

⚠ **Informational only** — not a medical diagnosis system.
""")

# PDF uploader
uploaded_files = st.file_uploader(
    "Upload trusted health PDFs (WHO, CDC, NIH, etc.)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    st.info(f"{len(uploaded_files)} PDF(s) uploaded. Processing...")

    # Extract text
    texts = []
    for pdf_file in uploaded_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

    # Chunk text
    chunk_size = 500
    overlap = 100
    chunks = []
    for text in texts:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap

    st.success(f"Created {len(chunks)} knowledge chunks.")

    # Embeddings
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.success("Vector database ready.")

    # Load LLM
    st.info("Loading language model (flan-t5-base)...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # ✅ Single text input for user query
    user_question = st.text_input("Ask a health-related question:")

    if user_question:
        # Retrieve relevant chunks
        query_embedding = embed_model.encode([user_question], convert_to_numpy=True)
        k = 3
        distances, indices = index.search(query_embedding, k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = " ".join(retrieved_chunks)

        # Create prompt
        prompt = f"""
        Answer this health question in a simple way anyone can understand.
        Context: {context}
        Question: {user_question}
        Answer:
        """

        # Call AI model
        result = llm(prompt, max_length=300, do_sample=False)

        # Display the AI answer
        st.subheader("🤖 AI Generated Answer")
        st.write(result[0]["generated_text"])
