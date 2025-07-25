import streamlit as st
import os
import shutil
import nest_asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
nest_asyncio.apply()
load_dotenv()
DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"
TEMP_UPLOAD_DIR = "temp_uploaded_data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
GEMINI_LLM_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
@st.cache_resource
def get_llm():
    """Initializes and returns the Google Gemini LLM."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, google_api_key=google_api_key)
        st.success(f"Connected to Google Gemini LLM: {GEMINI_LLM_MODEL}")
        return llm
    except Exception as e:
        st.error(f"Failed to connect to Google Gemini LLM '{GEMINI_LLM_MODEL}': {e}. Please check your GOOGLE_API_KEY and ensure the Generative Language API is enabled.")
        return None
@st.cache_resource
def get_embeddings():
    """Initializes and returns the Google Gemini embedding model."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=google_api_key)
        st.success(f"Using Google Gemini Embeddings: {GEMINI_EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize Google Gemini embeddings: {e}. Please check your GOOGLE_API_KEY and ensure the Generative Language API is enabled.")
        return None

@st.cache_resource
def load_documents_from_dir(directory):
    """Loads and splits documents from a given directory."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.startswith('.'):
            continue
        
        loader = None
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(filepath)
            except Exception as e:
                st.warning(f"Could not initialize PDF loader for '{filename}': {e}. Skipping.")
                continue

        if loader:
            try:
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            except Exception as e:
                st.warning(f"Error loading content from '{filename}': {e}")

    if not documents:
        return []

    chunks = text_splitter.split_documents(documents)
    return chunks

@st.cache_resource
def get_vector_store(_embeddings):
    """Creates or loads the FAISS vector store."""
    if _embeddings is None:
        return None

    chunks = load_documents_from_dir(DATA_DIR)
    
    if not chunks:
        st.warning("No documents found in 'data/' directory. Assistant will start with general knowledge. Upload documents in sidebar.")
        dummy_doc = Document(page_content="This is a placeholder document. Please upload your own files in the sidebar to ground the AI's knowledge!")
        vector_store = FAISS.from_documents([dummy_doc], _embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)
        return vector_store

    try:
        if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
            st.info("Loading existing FAISS index...")
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, _embeddings, allow_dangerous_deserialization=True)
            
            existing_doc_contents = {doc.page_content for doc in vector_store.docstore._dict.values()}
            new_chunks_to_add = [chunk for chunk in chunks if chunk.page_content not in existing_doc_contents]

            if new_chunks_to_add:
                st.info(f"Adding {len(new_chunks_to_add)} new document chunks to the vector store...")
                vector_store.add_documents(new_chunks_to_add)
                vector_store.save_local(FAISS_INDEX_DIR)
                st.success("Vector store updated with new documents!")
            else:
                st.info("No new documents to add from 'data/' directory. Using existing vector store.")
        else:
            st.info("Creating new FAISS index from documents in 'data/'...")
            vector_store = FAISS.from_documents(chunks, _embeddings)
            vector_store.save_local(FAISS_INDEX_DIR)
            st.success("FAISS index created!")
    except Exception as e:
        st.error(f"Error creating/loading FAISS index: {e}. Please check your documents and API key.")
        return None
    return vector_store

@st.cache_resource
def get_rag_chain(_llm, _retriever):
    """Constructs the RAG conversational chain."""
    if _llm is None or _retriever is None:
        st.warning("LLM or Retriever is not initialized. RAG chain cannot be created.")
        return None

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if necessary and otherwise return it as is.
    If the question is already standalone, just return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | _llm | StrOutputParser()

    qa_system_prompt = """You are a helpful AI assistant, like a copilot. You have access to user's documents and can answer questions based on the provided context:
    {context}

    If the question cannot be answered from the provided context, politely state that you don't have enough information in your documents to answer the question, but avoid making up answers.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_standalone_question_or_original(question: str, chat_history: list):
        if chat_history:
            return contextualize_q_chain.invoke({"input": question, "chat_history": chat_history})
        else:
            return question
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: _retriever.invoke(get_standalone_question_or_original(x["input"], x["chat_history"]))
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | qa_prompt
        | _llm
        | StrOutputParser()
    )
    return rag_chain
st.set_page_config(page_title="Streamlit Copilot Assistant", layout="wide")
st.title("👨‍💻 Streamlit Copilot Assistant")
st.markdown("Your intelligent AI companion, powered by your documents.")
llm = get_llm()
embeddings = get_embeddings()
vector_store = get_vector_store(embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) if vector_store else None
rag_chain = get_rag_chain(llm, retriever)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask your Copilot..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_chain:
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"An error occurred during AI response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again or check my configuration."})
    else:
        with st.chat_message("assistant"):
            st.warning("AI backend not fully initialized. Please ensure your GOOGLE_API_KEY is set and documents are processed.")
with st.sidebar:
    st.header("Your Data (Grounding)")
    st.write("Upload PDF or text files to ground the Copilot's knowledge.")
    uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True, key="file_uploader")

    if uploaded_files:
        st.info("Processing uploaded files...")
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        try:
            new_chunks = load_documents_from_dir(TEMP_UPLOAD_DIR)

            if new_chunks:
                if vector_store and embeddings:
                    st.info(f"Adding {len(new_chunks)} chunks from uploaded files to the knowledge base...")
                    vector_store.add_documents(new_chunks)
                    vector_store.save_local(FAISS_INDEX_DIR)
                    st.success("Uploaded documents processed and added to knowledge base!")
                else:
                    st.warning("Vector store or embeddings are not initialized. Cannot add uploaded documents.")
            else:
                st.info("No new content found in uploaded files to add.")
            st.cache_resource.clear()
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Knowledge base updated and chat reset. Please ask your questions!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to process uploaded files: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_DIR):
                shutil.rmtree(TEMP_UPLOAD_DIR)
            os.makedirs(TEMP_UPLOAD_DIR)
    st.markdown("---")
    st.markdown("### Reset Chat and Knowledge Base")
    if st.button("Clear Chat History", key="clear_chat_button"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    if st.button("Clear All Documents (and Chat)", key="clear_docs_button"):
        if os.path.exists(FAISS_INDEX_DIR):
            st.info("Clearing all documents from knowledge base...")
            shutil.rmtree(FAISS_INDEX_DIR)
            st.cache_resource.clear()
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("All documents cleared and knowledge base reset. Please upload new documents to begin.")
            st.rerun()
        else:
            st.info("No documents to clear from knowledge base.")
st.markdown("---")
st.caption("Built by your AI assistant")
