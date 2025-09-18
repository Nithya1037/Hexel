# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# ------------------ Custom CSS for Stylish UI ------------------
st.markdown("""
    <style>
    body, .stApp { background: var(--bg-color); }
    .main { background: var(--bg-color); }
    .chat-bubble-user {
        background: #1e88e5;
        color: white;
        padding: 12px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 8px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(30,136,229,0.08);
    }
    .chat-bubble-assistant {
        background: #f5f5f5;
        color: #222;
        padding: 12px;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 8px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .quick-action-btn {
        background: #e3f2fd;
        color: #1976d2;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
    }
    .quick-action-btn:hover {
        background: #bbdefb;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")   # Embedding model
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embedder, qa_model

embedder, qa_model = load_models()

# Granite models (text-generation)
@st.cache_resource
def load_granite():
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
    model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tok)
    return tok, model, pipe

granite_tokenizer, granite_model, granite_pipe = load_granite()

# ------------------ Helper Functions ------------------
def extract_text_from_pdf_bytes(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def create_chunks(text, chunk_size=400, overlap=50):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks, embedder):
    if len(chunks) == 0:
        return None
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=3):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    if q_vec.ndim == 1:
        q_vec = np.expand_dims(q_vec, axis=0)
    D, I = index.search(q_vec, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results

def answer_question(question, context, qa_model):
    if not context.strip():
        return "Sorry, I couldn’t find anything relevant in the PDF."
    result = qa_model(question=question, context=context)
    return result["answer"]

def summarize_with_granite(text, granite_pipe):
    """Summarize text chunks using Granite LLM."""
    prompt = f"Summarize the following academic text in simple terms:\n\n{text}\n\nSummary:"
    output = granite_pipe(prompt, max_new_tokens=100, do_sample=False)
    if isinstance(output, list):
        return output[0]["generated_text"]
    return str(output)

# ------------------ Streamlit UI ------------------

# ------------------ Theme Settings Sidebar ------------------
st.set_page_config(page_title="StudyMate 🤖", layout="centered")
with st.sidebar:
    st.header("🎨 Theme & Quick Actions")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
    st.markdown("---")
    if st.button("Show PDF Summary", key="summary_btn"):
        if "chunks" in st.session_state and st.session_state.chunks:
            summary = summarize_with_granite(" ".join(st.session_state.chunks[:3]), granite_pipe)
            st.session_state.messages.append({"role": "assistant", "content": f"📝 PDF Summary: {summary}"})
            st.rerun()
        else:
            st.warning("No PDF uploaded yet.")
    if st.button("Clear Chat (Quick)", key="quick_clear"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.info("Switch theme for a different look. Use quick actions for fast results!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []


# ------------------ Theme Color Logic ------------------
if theme == "Dark":
    st.markdown("<style>:root { --bg-color: #181818; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>:root { --bg-color: #f7f9fa; }</style>", unsafe_allow_html=True)

st.title("📚 StudyMate – PDF Chatbot with Granite LLM")

uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf_bytes(uploaded_file)
        chunks = create_chunks(text, chunk_size=450, overlap=60)
        if not chunks:
            st.error("No text found in this PDF.")
        else:
            index = build_faiss_index(chunks, embedder)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.success(f"✅ PDF processed successfully! Extracted {len(chunks)} chunks.")

# Display chat history

# ------------------ Stylish Chat History ------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-assistant'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("💬 Ask me anything about your PDF...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.spinner("🤖 Thinking..."):
        if not st.session_state.index:
            answer = "⚠️ Please upload a PDF first."
        else:
            top_chunks = retrieve_relevant_chunks(user_query, st.session_state.index, st.session_state.chunks, embedder)
            context = " ".join(top_chunks)
            answer = answer_question(user_query, context, qa_model)

            # Extra: Granite summarization of retrieved context
            granite_summary = summarize_with_granite(context, granite_pipe)
            answer = f"{answer}\n\n📝 *Granite Summary of context:* {granite_summary}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ------------------ Main Clear Chat Button ------------------
if st.button("🗑️ Clear Chat", help="Clear all chat history"):
    st.session_state.messages = []
    st.rerun()

# ------------------ Hugging Face Granite Demo ------------------
st.markdown("---")
st.subheader("🧪 Hugging Face Granite Demo")

hf_messages = [{"role": "user", "content": "Who are you?"}]
resp = granite_pipe("Who are you?", max_new_tokens=40)
st.write("**Granite Pipeline Output:**", resp[0]["generated_text"] if isinstance(resp, list) else resp)

inputs = granite_tokenizer.apply_chat_template(
    hf_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(granite_model.device)

outputs = granite_model.generate(**inputs, max_new_tokens=40)
decoded = granite_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
st.write("**Granite Direct Model Output:**", decoded)
