from loaders import load_documents
from embeddings import build_faiss_index, load_faiss_index
from dspy_pipeline import build_faiss_index_with_dspy
from chains import build_rag_chain
import os
import shutil
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_qa_chain(
    use_dspy: bool = False,
    rebuild_index: bool = False,
    model_name: str = "gpt-4o-mini"
) -> Optional[Any]:
    """Initialize the QA chain with configurable options.
    
    Args:
        use_dspy (bool): Whether to use DSPy processing
        rebuild_index (bool): Force rebuild of the index
        model_name (str): Name of the OpenAI model to use
        
    Returns:
        Optional[Any]: Initialized QA chain or None if initialization fails
    """
    try:
        # 1. Configure directories
        DATA_DIR = "data"
        INDEX_DIR = "models/faiss_index"
        
        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

        # 2. Debug: Print directory contents
        logger.info(f"Checking contents of {DATA_DIR}...")
        if os.path.exists(DATA_DIR):
            files = os.listdir(DATA_DIR)
            logger.info(f"Found files: {files}")
        else:
            logger.error("DATA_DIR does not exist!")
            return None

        # 3. Try to load existing index
        vectorstore = None
        if not rebuild_index and os.path.exists(INDEX_DIR):
            logger.info("Attempting to load existing FAISS index...")
            vectorstore = load_faiss_index(INDEX_DIR)
            if vectorstore:
                logger.info("Successfully loaded existing FAISS index")

        # 4. Build new index if needed
        if vectorstore is None:
            logger.info("Building new FAISS index...")
            # Clear existing index
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            os.makedirs(INDEX_DIR)
            
            # Load and process documents
            documents = load_documents(DATA_DIR, use_dspy=use_dspy)
            if not documents:
                logger.error("No documents were loaded!")
                return None
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Build index with or without DSPy
            if use_dspy:
                vectorstore = build_faiss_index_with_dspy(
                    documents, 
                    INDEX_DIR,
                    model_name=model_name
                )
            else:
                vectorstore = build_faiss_index(documents, INDEX_DIR)
            
            if not vectorstore:
                logger.error("Failed to build vector store!")
                return None
            
            logger.info("Successfully built new FAISS index")

        # 5. Build and test chain
        chain = build_rag_chain(
            vectorstore,
            model_name=model_name
        )
        
        if chain:
            logger.info("Successfully initialized QA chain")
            return chain
        else:
            logger.error("Failed to initialize QA chain")
            return None

    except Exception as e:
        logger.error(f"Error initializing QA chain: {e}")
        return None

def check_system_health():
    """Verify system components and configurations."""
    status = {
        "api_key": False,
        "directories": False,
        "document_loading": False,
        "vectorstore": False,
        "chain": False,
        "dspy": False
    }
    
    try:
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        status["api_key"] = bool(api_key)
        
        # Check directories
        DATA_DIR = "data"
        INDEX_DIR = "models/faiss_index"
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        status["directories"] = os.path.exists(DATA_DIR) and os.path.exists(INDEX_DIR)
        
        # Check document loading
        test_doc = "This is a test document."
        with open(os.path.join(DATA_DIR, "test.txt"), "w") as f:
            f.write(test_doc)
        docs = load_documents(DATA_DIR)
        status["document_loading"] = len(docs) > 0
        
        # Check vectorstore
        vectorstore = build_faiss_index(docs, INDEX_DIR)
        status["vectorstore"] = vectorstore is not None
        
        # Check chain
        if vectorstore:
            chain = build_rag_chain(vectorstore)
            test_response = chain({"query": "test"})
            status["chain"] = "result" in test_response
        
        # Check DSPy
        try:
            from dspy_pipeline import IEPPipeline
            pipeline = IEPPipeline()
            status["dspy"] = True
        except:
            status["dspy"] = False
            
        # Cleanup test file
        os.remove(os.path.join(DATA_DIR, "test.txt"))
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
    
    return status

if __name__ == "__main__":
    # Initialize chain with configurable options
    qa_chain = initialize_qa_chain(
        use_dspy=False,  # Set to True to enable DSPy processing
        rebuild_index=False,  # Set to True to force index rebuild
        model_name="gpt-4o-mini"
    )
    
    if qa_chain:
        logger.info("System initialized successfully")
        
        # Optional: Add interactive testing
        while True:
            query = input("\nEnter a question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            try:
                response = qa_chain({"query": query})
                print("\nAnswer:", response['result'])
                print("\nSources:")
                for doc in response.get('source_documents', []):
                    print("-" * 40)
                    print(doc.page_content[:200] + "...")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
    else:
        logger.error("Failed to initialize system")