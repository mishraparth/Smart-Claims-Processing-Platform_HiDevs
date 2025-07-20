import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Smart Claims Processing", layout="wide")
st.title("📄 Smart Claims Processing Platform")
st.write("Upload your insurance policy documents, and ask questions about claims to get instant, AI-powered answers.")

# --- Global Variables & Constants ---
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore_db"

# --- Helper Functions ---

# Use Streamlit's cache to avoid re-processing on every interaction
@st.cache_resource
def process_and_store_documents(data_path):
    """Loads, splits, embeds, and stores documents in a Chroma vector store."""
    if not os.path.exists(data_path):
        st.error(f"Data directory not found at: {data_path}")
        return None

    all_docs = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, filename))
            all_docs.extend(loader.load())

    if not all_docs:
        st.warning("No PDF documents found in the 'data' directory.")
        return None

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=VECTORSTORE_DIR
    )
    
    st.success(f"✅ Documents processed and stored successfully! {len(splits)} chunks created.")
    return vectorstore

def get_qa_chain():
    """Builds the RetrievalQA chain for answering questions."""
    # Load the persisted vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Initialize the LLM
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# --- Main Application Logic ---

# Check for Groq API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found! Please add it to your .env file.")
else:
    # Button to process documents
    if st.button("Process Policy Documents", key="process_docs"):
        with st.spinner("Processing documents... This might take a moment."):
            process_and_store_documents(DATA_DIR)

    st.header("Ask a Question About a Claim")
    user_question = st.text_input("e.g., Is water damage from a leaky roof covered under my policy?", key="question_input")

    if user_question:
        if not os.path.exists(VECTORSTORE_DIR):
            st.warning("Please process the documents first before asking a question.")
        else:
            with st.spinner("Finding an answer..."):
                try:
                    qa_chain = get_qa_chain()
                    result = qa_chain({"query": user_question})
                    
                    st.subheader("📝 Answer:")
                    st.write(result["result"])

                    with st.expander("📚 See Relevant Policy Sections"):
                        for doc in result["source_documents"]:
                            st.info(f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}")
                            st.text(doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred: {e}")