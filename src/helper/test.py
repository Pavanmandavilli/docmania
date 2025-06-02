import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

token=os.getenv("HUGGINGFACEHUB_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token

def process_documents(uploaded_files, save_path="./files/", persist_directory=None):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    loader = DirectoryLoader(save_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context:\n{context}\n\nAnswer the following question:\n{question}"
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_docs_chain
    )

    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    def wrap_line(line, width):
        words = line.split()
        if not words:
            return line
        wrapped_lines = []
        current_line = words[0]
        for word in words[1:]:
            if len(current_line) + len(word) + 1 <= width:
                current_line += ' ' + word
            else:
                wrapped_lines.append(current_line)
                current_line = word
        wrapped_lines.append(current_line)
        return '\n'.join(wrapped_lines)

    lines = text.split('\n')
    wrapped_lines = [wrap_line(line, width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])



