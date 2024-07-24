import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

working_dir = os.path.dirname(os.path.abspath(__file__))
llm = Ollama(
    model="llama3:latest",
    temperature=0.7
)

embeddings = HuggingFaceEmbeddings()


def get_answer(file_name, query):
    file_path = f"{working_dir}/{file_name}"
    #   loading the document
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    #   create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)

    text_chunks = text_splitter.split_documents(documents)
    #    vector embeddings from text chunk
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    response = qa_chain.invoke({"query": query})

    return response["result"]
