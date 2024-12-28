from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os

def build_rag_chain(vectorstore):
    """Build an optimized RAG chain with GPT-4o Mini."""
    
    # Debug: Test vectorstore
    test_results = vectorstore.similarity_search("test", k=1)
    print(f"Vectorstore test retrieval: {len(test_results)} results")
    
    # 1. Configure OpenAI LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
        openai_api_key=api_key
    )
    
    # 2. Create prompt template
    prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # 3. Build chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "score_threshold": None  # Remove threshold
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": True
        }
    )
    
    # 4. Test the chain
    test_response = chain({"query": "test"})
    print("Chain test response:", test_response.keys())
    
    return chain
