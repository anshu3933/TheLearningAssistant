import streamlit as st
from chains import build_rag_chain
from embeddings import build_faiss_index, load_faiss_index
from loaders import load_documents
import os
import tempfile
import shutil
import time
from streamlit_feedback import streamlit_feedback

# Page Configuration
st.set_page_config(
    page_title="Educational Assistant",
    page_icon=":books:",
    layout="wide"
)

# Initialize directories
DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize session state
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_feedback(feedback, response_text):
    """Process feedback submitted by users."""
    score = feedback["score"]
    text = feedback.get("text", "")
    print(f"Feedback received - Score: {score}, Text: {text}")
    print(f"For response: {response_text}")

    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []

    st.session_state.feedback_history.append({
        "score": score,
        "text": text,
        "response": response_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    return True

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and update the RAG chain."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Debug print
        print(f"Processing file: {uploaded_file.name}")
        
        # Load documents
        documents = load_documents(temp_dir)
        if not documents:
            st.error("No text could be extracted from the uploaded file.")
            return False
        
        # Debug print
        print(f"Loaded {len(documents)} documents")
        print(f"First document content: {documents[0].page_content[:200]}")
        
        # Build index
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR)
        
        vectorstore = build_faiss_index(documents, INDEX_DIR)
        
        # Test retrieval
        test_results = vectorstore.similarity_search("test", k=1)
        print(f"Test retrieval found {len(test_results)} documents")
        
        st.session_state["chain"] = build_rag_chain(vectorstore)
        return True

st.title("Educational Assistant with GPT-4o Mini")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("Document Upload")
    api_key = st.text_input("OpenAI API Key", type="password")

    if not api_key:
        st.error("Please provide your OpenAI API key.")

    uploaded_files = st.file_uploader(
        "Upload educational documents",
        type=["txt", "docx", "doc", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state["documents_processed"]:
        with st.spinner("Processing documents..."):
            success = all(process_uploaded_file(file) for file in uploaded_files)
            if success:
                st.session_state["documents_processed"] = True
                st.success("Documents processed successfully!")
            else:
                st.error("Error processing some documents.")

    if st.session_state["documents_processed"]:
        if st.button("Clear Documents"):
            st.session_state["documents_processed"] = False
            st.session_state["chain"] = None
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            st.experimental_rerun()

# Create tabs
tab1, tab2 = st.tabs(["Document Q&A", "Chat"])

# Document Q&A Tab
with tab1:
    st.header("Document Q&A")

    query = st.text_area(
        "Ask a question about the documents:",
        placeholder="Example: Can you summarize the main points?",
        disabled=not st.session_state["documents_processed"]
    )

    if query:
        if st.session_state["chain"]:
            with st.spinner("Generating response..."):
                try:
                    start_time = time.time()
                    
                    # Get response
                    chain_response = st.session_state["chain"]({"query": query})
                    end_time = time.time()
                    
                    # Extract answer and sources
                    answer = chain_response.get('result', '')
                    sources = chain_response.get('source_documents', [])
                    
                    # Display response
                    st.write("### Response")
                    st.write(answer)
                    
                    # Display sources
                    with st.expander("View Retrieved Context"):
                        if sources:
                            for i, doc in enumerate(sources, 1):
                                st.write(f"Source {i}:")
                                st.write(doc.page_content)
                                st.write("---")
                        else:
                            st.write("No source documents were retrieved.")
                            # Debug info
                            st.write("Debug Info:")
                            st.write(f"Chain type: {type(st.session_state['chain'])}")
                            st.write(f"Response type: {type(chain_response)}")
                            st.write(f"Response keys: {chain_response.keys() if isinstance(chain_response, dict) else 'Not a dict'}")
                    
                    st.write(f"Response time: {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error("Full error:", exc_info=True)
        else:
            st.warning("Please upload documents first!")

# Chat Interface Tab
with tab2:
    st.header("Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.get("documents_processed"):
            st.error("Please upload documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Use the call method instead of run
                        chain_response = st.session_state["chain"]({"query": prompt})
                        response = chain_response.get('result', '')
                        sources = chain_response.get('source_documents', [])
                        
                        # Add response to messages
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                        
                        # Show sources in expander
                        with st.expander("View Source Context"):
                            if sources:
                                for i, doc in enumerate(sources, 1):
                                    st.write(f"Source {i}:")
                                    st.write(doc.page_content)
                                    st.write("---")
                            else:
                                st.write("No source documents were retrieved.")
                                
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.error(error_message)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Add feedback history viewer in sidebar
with st.sidebar:
    if st.session_state.get("feedback_history"):
        st.subheader("Feedback History")
        if st.button("Clear Feedback History"):
            st.session_state.feedback_history = []
            st.experimental_rerun()

        for feedback in st.session_state.feedback_history:
            with st.expander(f"Feedback from {feedback['timestamp']}"):
                st.write(f"Score: {feedback['score']}")
                if feedback['text']:
                    st.write(f"Comment: {feedback['text']}")
                st.write("Response:", feedback['response'])

# Footer
st.markdown("---")
st.markdown("Educational Assistant")
