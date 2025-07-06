from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

#Load the PDF files from the directory
Data_path = ".venv\Data"
def load_pdf_file(data):
    Loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=Loader.load()
    return documents

Documents=load_pdf_file(Data_path)
print(f"Number of documents loaded: {len(Documents)}")

#create Chunks of text from the documents
def craete_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks=craete_chunks(extracted_data=Documents)
print(f"Number of text chunks created: {len(text_chunks)}")

#Create embeddings for the text chunks
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",)
    
    return embedding_model
embedding_model=get_embedding_model()

#Create a vector store from the text chunks and embeddings Faiss
db_faiss_path="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(db_faiss_path)
print(f"Vector store saved at: {db_faiss_path}")


