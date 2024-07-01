
from langchain_community.llms import ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
import streamlit as st
from constant import cohere_key

embeddings_model = CohereEmbeddings(cohere_api_key=cohere_key)

llm=Cohere(cohere_api_key=cohere_key)

def doc_preprocessing():
    loader = PyPDFLoader('med.pdf')
             
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, 
        chunk_overlap=200
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
   
    
    docs_split = doc_preprocessing()
    doc_db = Chroma.from_documents(documents=docs_split, embedding=embeddings_model)
    return doc_db

doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result

def main():
    st.title("Medical ChatBot")
    st.write("Your trusted companion for all medical queries, providing accurate and reliable information around the clock.")
    text_input = st.text_input("Enter Your Question") 
    if st.button("Submit"):
        if len(text_input)>0:
            st.info("Question " + text_input)
            answer = retrieval_answer("Answer"+ text_input)
            st.success(answer)

            

if __name__ == "__main__":
    main()