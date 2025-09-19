"""
Offline SLM Chatbot (app.py)
--------------------------------
A simple Streamlit app demonstrating an offline chatbot using a Small Language Model (SLM).
This file is intentionally documented with DETAILED comments so you (or a recruiter) can easily
understand each step. Keep in mind: model names used here are examples — you can replace them
with any compatible causal LM from Hugging Face.

How to run:
1. Install dependencies: pip install -r requirements.txt
2. Run: streamlit run app.py

Notes:
- This app tries to be CPU friendly but will use CUDA if available.
- If a model download is large, it may take time on first run.
"""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import os

# LangChain + FAISS imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# ---------------------------
# Configuration / Defaults
# ---------------------------
DEFAULT_MODEL = "microsoft/phi-2"  # SLM example — replace if needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast embedding model

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_TOKENS = 150

# ---------------------------
# Model loader (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_name: str, device: str = DEFAULT_DEVICE):
    """
     Load tokenizer and model from Hugging Face.
    - model_name: HF model identifier (e.g., "microsoft/phi-2")
    - device: "cpu" or "cuda"
    Returns tuple (tokenizer, model, device)
    """
    st.info(f"Loading generator model `{model_name}` on `{device}`. This may take time on first run.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Load and return a HuggingFaceEmbeddings object from LangChain which
    internally uses sentence-transformers models. Cached for speed.
    """
    st.info(f"Loading embedding model `{model_name}`.")
    emb = HuggingFaceEmbeddings(model_name=model_name)
    return emb

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="SLM Document Q&A", layout="wide")
st.title("📄 Document Q&A with SLM (Small Language Model)")

# Sidebar - model & generation settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Generator model (HF)", value=DEFAULT_MODEL)
    device_choice = st.selectbox("Device", options=["cuda", "cpu"], index=0 if DEFAULT_DEVICE=="cuda" else 1)
    max_new_tokens = st.number_input("Max new tokens", min_value=16, max_value=512, value=DEFAULT_MAX_TOKENS, step=16)
    k_results = st.number_input("Top-k documents to search", min_value=1, max_value=10, value=3, step=1)
    clear_cache = st.button("Clear cached vector DB")

# Load generator model (tokenizer + model)
try:
    tokenizer, model, model_device = load_generator(model_name, device_choice)
except Exception as e:
    st.error(f"Failed to load generator model `{model_name}`. Error: {e}")
    st.stop()

# Load embeddings (cached)
embeddings = get_embeddings()

# Uploaded PDF handler
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# We'll cache the FAISS index per uploaded file content using Streamlit's session_state
if uploaded_file:
    # Save uploaded file to temp path so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Use file_key to uniquely identify file in session cache
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"
    if "vector_dbs" not in st.session_state:
        st.session_state.vector_dbs = {}

    if clear_cache and file_key in st.session_state.vector_dbs:
        del st.session_state.vector_dbs[file_key]
        st.success("Cleared cached vector DB for this file.")

    if file_key not in st.session_state.vector_dbs:
        with st.spinner("Loading and splitting PDF..."):
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(docs)

        with st.spinner("Creating vector index (FAISS)..."):
            db = FAISS.from_documents(chunks, embeddings)

        st.session_state.vector_dbs[file_key] = db
        st.success("Indexed document and stored vector DB in session cache.")
    else:
        db = st.session_state.vector_dbs[file_key]
        st.info("Using cached vector DB for uploaded file.")

    # Input for user query
    query = st.text_input("Ask a question from the uploaded document:", key=f"q-{file_key}")

    if query:
        with st.spinner("Searching for relevant document chunks..."):
            results = db.similarity_search(query, k=int(k_results))

        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

        with st.spinner("Generating answer from the SLM..."):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()

        st.subheader("Answer")
        st.write(answer)

        # ---------------------------
        # Helpful notes for the user (explainability)
        # ---------------------------
        st.markdown("---")
        st.subheader("Notes & Tips")
        st.markdown("""
        - **Offline**: This app runs **locally** and does not send your data to external APIs.
        - **Replace model**: If `microsoft/phi-2` is not available or too large, change the model name in the sidebar to a smaller model such as `distilgpt2` or `EleutherAI/gpt-neo-125M`.
        - **Performance**: Small models run fast on CPU; larger models may require more RAM or GPU.
        - **Improvements**: For a better chat experience, you can add context-window management (prepend recent messages to the prompt) or fine-tune a small model on conversational data.
        """)
        st.write("**Retrieved context (for debugging / transparency):**")
        for i, doc in enumerate(results, 1):
            st.write(f"**Chunk {i}:**")
            st.write(doc.page_content[:1000])  # show snippet

    # cleanup temp file after use
    os.unlink(temp_path)
else:
    st.info("Upload a PDF to enable Document Q&A. The app will index the document and allow semantic search.")
