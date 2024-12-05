from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf_file(path):
    loader = DirectoryLoader(path,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    data = loader.load()
    return data


def text_spliter(data):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_spliter.split_documents(data)
    return text_chunks

def load_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings