from src.helper import load_pdf_file, text_spliter, load_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

data=load_pdf_file(data='Data/')
documents=text_spliter(data)
embeddings = load_huggingface_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name = index_name,
    embedding=embeddings
)