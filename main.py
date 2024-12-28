from loaders import load_documents
from embeddings import build_faiss_index, load_faiss_index
from chains import build_rag_chain
import os
import shutil

def initialize_qa_chain():
    DATA_DIR = "data"
    INDEX_DIR = "models/faiss_index"

    # Ensure the index directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Debug: Print contents of DATA_DIR
    print(f"Checking contents of {DATA_DIR}...")
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print(f"Found files: {files}")
    else:
        print("DATA_DIR does not exist!")

    # Try to load existing index, rebuild if there's any issue
    vectorstore = None
    if os.path.exists(INDEX_DIR):
        print("Attempting to load FAISS index...")
        vectorstore = load_faiss_index(INDEX_DIR)
        if vectorstore:
            print("Successfully loaded FAISS index")

    # If loading failed or index doesn't exist, build new index
    if vectorstore is None:
        print("Building new FAISS index...")
        # Clear the index directory if it exists
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR)
        
        documents = load_documents(DATA_DIR)
        print(f"Loaded {len(documents)} documents")  # Debug print
        if documents:
            vectorstore = build_faiss_index(documents, INDEX_DIR)
            print("Built new FAISS index")
        else:
            print("No documents were loaded!")
            return None

    # Build and return the chain
    chain = build_rag_chain(vectorstore)
    return chain

# Initialize the chain
qa_chain = initialize_qa_chain()
if qa_chain:
    print("Chain initialized successfully")
else:
    print("Failed to initialize chain")