import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


#cohere_api_key= "rCTqOlfaNwEuTCO8ALXYryaAoBDmH8Yky6LncQnO"

def main():
    cohere_api_key= "rCTqOlfaNwEuTCO8ALXYryaAoBDmH8Yky6LncQnO"
    model_id = "damo/nlp_corom_sentence-embedding_english-base"
    st.set_page_config(page_title="Document Question Answer Chatbot")
    st.header("Ask anything from your PDF")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
      
      #st.write(text)
      
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)

      #st.write(chunks)
    if st.checkbox("Cohere Embeddings"):
        st.write("Cohere Embeddings Selected!")
        embeddings_cohere = CohereEmbeddings(model= "embed-english-light-v2.0",cohere_api_key=cohere_api_key)
        context = FAISS.from_texts(chunks, embeddings_cohere)

            # show user input
        query = st.text_input("Ask a question about your PDF:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)

    elif st.checkbox("Model Scope Embeddings"):
        model_id = "damo/nlp_corom_sentence-embedding_english-base"
        st.write("Model Scope Embeddings Selected!")
        embeddings_model_scope = ModelScopeEmbeddings(model_id = model_id)
        context = FAISS.from_texts(chunks, embeddings_model_scope)

            # show user input
        query = st.text_input("Ask a question about your PDF:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)

    elif st.checkbox("Sentence Transformers Embeddings"):
        st.write("Sentence Transformer Embeddings Selected!")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_st = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        context = FAISS.from_texts(chunks, embeddings_st)

            # show user input
        query = st.text_input("Ask a question about your PDF:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)

    elif st.checkbox("JINA Ai Embeddings"):
        st.write("JINA Ai Embeddings Selected!")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_st = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        context = FAISS.from_texts(chunks, embeddings_st)

            # show user input
        query = st.text_input("Ask a question about your PDF:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)




    
if __name__ == '__main__':
    main()