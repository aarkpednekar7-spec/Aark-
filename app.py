import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import tempfile

def load_document(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(tmp_file_path)
    else:
        st.error("Unsupported file type.")
        os.remove(tmp_file_path) # Clean up temp file on error
        return None

    try:
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None
    finally:
        # Ensure the temporary file is deleted
        os.remove(tmp_file_path)

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits, api_key):
    print(f"DEBUG: Using API key for embeddings: {api_key[:5]}...{api_key[-5:]}")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    print(f"DEBUG: Using API key for chat model: {api_key[:5]}...{api_key[-5:]}")
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="RAG Application")
    st.header("RAG Application")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    if st.button("Validate API Key"):
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Attempt a simple API call to validate the key
                client.models.list()
                st.success("API Key is valid!")
            except Exception as e:
                st.error(f"Invalid API Key: {e}")
        else:
            st.error("Please enter an API Key.")

    uploaded_file = st.file_uploader("Upload your document (PDF, TXT):", type=["pdf", "txt"])

    if st.button("Process Document"):
        if uploaded_file and api_key:
            try:
                os.environ["OPENAI_API_KEY"] = api_key  # Set API key for Langchain
            except Exception as e:
                st.error(f"Error setting API key: {e}")
                return

            with st.spinner("Processing document..."):
                try:
                    # Load document
                    documents = load_document(uploaded_file)
                    if documents:
                        # Split documents
                        splits = split_documents(documents)
                        # Create vector store
                        vectorstore = create_vector_store(splits, api_key)
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                        st.success("Document processed and RAG model ready!")
                    else:
                        st.error("Failed to load document.")
                except Exception as e:
                    st.error(f"Error during document processing: {e}")
        elif not api_key:
            st.error("Please enter your OpenAI API Key first.")
        else:
            st.error("Please upload a document first.")

    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write("User:", message.content)
                else:
                    st.write("AI:", message.content)
        else:
            st.error("Please process a document first.")

if __name__ == "__main__":
    main()
