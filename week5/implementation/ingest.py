import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

MODEL = 'gpt-4.1-nano'

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')

def fetch_documents():

    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob = "**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding' : 'utf-8'})
        docs = loader.load()
        for doc in docs:
            doc.metadata['doc_type'] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embedding(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
    collection = vector_store._collection
    sample_embedding = collection.get(limit=1, include=['embeddings'])['embeddings'][0]
    dimension = len(sample_embedding)
    print(f"There are {collection.count()} vectors having {dimension} dimensions")
    return vector_store

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embedding(chunks)
    print("Ingestion completed")