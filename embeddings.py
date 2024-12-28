from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import os

def build_faiss_index(documents, persist_directory):
    """Build an optimized FAISS index."""
    print(f"Building index from {len(documents)} documents")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    if texts:
        print("Sample chunk:", texts[0].page_content[:200])
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(persist_directory)
    
    # Test retrieval
    results = vectorstore.similarity_search("test", k=1)
    print(f"Test retrieval: {len(results)} results")
    
    return vectorstore

def load_faiss_index(persist_directory):
    """Load the FAISS index."""
    api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key
    )
    return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

    return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)  # Only use this if you trust the source of the index)

