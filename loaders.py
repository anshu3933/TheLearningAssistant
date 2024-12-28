import os
from langchain.schema import Document as LangchainDocument
from PyPDF2 import PdfReader
from docx import Document

def load_documents(data_path):
    """Load documents and return LangChain-compatible documents."""
    docs = []
    print(f"Scanning directory: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Directory does not exist: {data_path}")
        return docs

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        print(f"Processing file: {file_path}")
        
        try:
            if file.endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                print(f"Extracted {len(text)} characters from DOCX")
                docs.append(LangchainDocument(page_content=text, metadata={"source": file}))
            
            elif file.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                print(f"Extracted {len(text)} characters from PDF")
                docs.append(LangchainDocument(page_content=text, metadata={"source": file}))
            
            elif file.endswith(".txt"):
                with open(file_path, "r", encoding='utf-8') as file_content:
                    text = file_content.read()
                    print(f"Extracted {len(text)} characters from TXT")
                    docs.append(LangchainDocument(page_content=text, metadata={"source": file}))
            
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    print(f"Total documents loaded: {len(docs)}")
    return docs