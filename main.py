import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from pymilvus import connections
from langchain_milvus import Milvus
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

load_dotenv()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
MILVUS_URI = st.secrets["MILVUS_URI"]
MILVUS_TOKEN = st.secrets["MILVUS_TOKEN"]
MILVUS_COLLECTION = st.secrets["MILVUS_COLLECTION"]

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="DocMania",
    page_icon="🚀",
    layout="wide"
)


st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #FF6B6B;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}


.chat-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 25px;
    color: black !important;
    line-height: 1.7;
}

.source-box {
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 8px;
    margin-top: 20px;
    margin-bottom: 20px;
    color: black !important;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "models/gemini-embedding-001"

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model=self.model,
                content=text
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model=self.model,
            content=text
        )
        return response["embedding"]

connections.connect(
    alias="default",
    uri=MILVUS_URI,
    token=MILVUS_TOKEN
)

@st.cache_resource
def load_vector_db():
    embedding = GeminiEmbeddings(GEMINI_API_KEY)

    vectordb = Milvus(
        collection_name=MILVUS_COLLECTION,
        connection_args={
            "uri": MILVUS_URI,
            "token": MILVUS_TOKEN
        },
        embedding_function=embedding,
        auto_id=True,
        drop_old=False
    )

    return vectordb

def process_documents(uploaded_files):
    vectordb = load_vector_db()
    os.makedirs("files", exist_ok=True)

    all_docs = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join("files", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(documents)
        all_docs.extend(split_docs)

    vectordb.add_documents(all_docs)


def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not found, say "Not found in document."

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text


st.markdown('<p class="main-title">🚀 DocMania </p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Navigation")
mode = st.sidebar.radio(
    "Choose Mode",
    ["📂 Document Search", "🌍 Global Search"]
)


if mode == "📂 Document Search":

    st.markdown('<p class="section-title">Upload & Search Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Process Documents"):
            with st.spinner("Processing and indexing documents..."):
                process_documents(uploaded_files)
            st.success("✅ Documents indexed successfully!")

        question = st.text_input("🔎 Ask a question from uploaded documents")

        if question:
            with st.spinner("Searching and generating answer..."):
                vectordb = load_vector_db()
                docs = vectordb.similarity_search(question, k=3)
                answer = generate_answer(question, docs)

            st.markdown("### 🤖 Answer")
            st.markdown(f'<div class="chat-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("### 📄 Sources")
            for i, doc in enumerate(docs):
                st.markdown(
                    f'<div class="source-box"><b>Source {i+1}:</b><br>{doc.page_content[:500]}...</div>',
                    unsafe_allow_html=True
                )

    else:
        st.warning("⚠️ Please upload documents to enable search.")


elif mode == "🌍 Global Search":

    st.markdown('<p class="section-title">Search Across Entire Database</p>', unsafe_allow_html=True)

    question = st.text_input("🔎 Ask a global question")

    if question:
        with st.spinner("Searching entire database..."):
            vectordb = load_vector_db()
            docs = vectordb.similarity_search(question, k=5)
            answer = generate_answer(question, docs)

        st.markdown("### 🌍 Global Answer")
        st.markdown(f'<div class="chat-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown("### 📄 Sources")
        for i, doc in enumerate(docs):
            st.markdown(
                f'<div class="source-box"><b>Source {i+1}:</b><br>{doc.page_content[:500]}...</div>',
                unsafe_allow_html=True
            )
