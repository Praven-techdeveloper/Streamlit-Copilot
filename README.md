# Streamlit-Copilot
Key Features
Document Grounding: Upload PDF and text files to customize the AI's knowledge

Conversational Memory: Maintains context across multiple questions

Vector Search: FAISS-based retrieval finds relevant document passages

Google Gemini Integration: Uses cutting-edge Gemini models for reasoning

Streamlit UI: Clean, intuitive interface with sidebar controls

Persistent Knowledge Base: Saves document embeddings between sessions

Setup Instructions
Clone the repository:

bash
git clone https://github.com/yourusername/streamlit-copilot-assistant.git
cd streamlit-copilot-assistant
Install dependencies:

bash
pip install -r requirements.txt
Set up environment variables:

Create a .env file in the project root

Add your Google API key:

text
GOOGLE_API_KEY=your_api_key_here
Enable the Generative Language API in Google Cloud Console

Prepare document directory:

bash
mkdir data
# Add your PDF/text files to this directory for initial knowledge base
Running the Application
Start the Streamlit app:

bash
streamlit run app.py
The application will automatically:

Load your API key from .env

Process documents in the data/ directory

Create a FAISS vector store for semantic search

Launch a web interface at localhost:8501

Usage Guide
Upload documents:

Use the sidebar to upload PDF or text files

The assistant will process and add them to its knowledge base

Ask questions:

Type your question in the chat input at the bottom

The assistant will retrieve relevant passages and generate answers

Manage knowledge:

Clear chat history without affecting documents

Reset the entire knowledge base including uploaded documents

Add new documents at any time

Technical Stack
Framework: Streamlit

LLM: Google Gemini 1.5 Flash

Embeddings: Google Gemini Embedding-001

Vector Store: FAISS

Document Processing: PyPDF, LangChain TextLoader

Caching: Streamlit cache_resource

Troubleshooting
If you encounter issues:

Verify your Google API key is valid and has Generative Language API enabled

Check that documents are in supported formats (PDF or text)

Ensure files don't have special characters in names

Delete the faiss_index directory to reset the knowledge base

Check terminal for error messages when running the app
