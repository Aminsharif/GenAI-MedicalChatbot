import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from flask import Flask, render_template, jsonify, request
from src.helper import load_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')


embeddings = load_huggingface_embeddings()
llm = ChatGroq(model_name = "llama3-8b-8192",temperature=0.5,max_tokens=500, groq_api_key = os.getenv('GROQ_API_KEY'))

index_name = 'medicalbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    print('........................')
    response = rag_chain.invoke({"input": msg})
    print('response: ', response['answer'])
    print('_________________________________')
    return str(response['answer'])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)