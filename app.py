import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st 
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnableMap
from langchain_core.output_parsers import StrOutputParser
    
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


VECTOR_STORE_DIR = "FAISS_store"
if not os.path.exists(VECTOR_STORE_DIR):    
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
faiss_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")
pkl_file = os.path.join(VECTOR_STORE_DIR, "index.pkl")

if not os.path.exists(faiss_file) and not os.path.exists(pkl_file):
    print("Creating a new FAISS store...")
    docs = ["Python is a programming language.", 
            "FAISS is a library for efficient similarity search."]
    faiss_store = FAISS.from_texts(docs, embedding=embedding_model)
    faiss_store.save_local(VECTOR_STORE_DIR)    
else:
    faiss_store = FAISS.load_local(
        VECTOR_STORE_DIR,  # folder where index is saved
        embedding_model,
        allow_dangerous_deserialization=True)


def get_text_from_pdf(file):
    """Extract text from a PDF file."""
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def split_text(text):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    return texts

def store_vector_from_text(text, file_name, doc_id):
    """store a vector from the text chunks."""
    splited_text = split_text(text)
    new_docs = [
        Document(page_content = chunk, metadata={"source": file_name, "doc_id": f"{doc_id}_chunk_{idx}"}) for idx, chunk in enumerate(splited_text)    
    ]
    faiss_store.add_documents(new_docs)
    faiss_store.save_local("FAISS_store") 

def retrive_context(query, k=3):
    """Retrieve context from the vector store based on the query."""
    docs = faiss_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Legal Document Review Application", layout="wide")
    st.title("Legal Document Review Application")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Legal Querstion")
        legal_query = st.text_area("Enter Legal Queries", height=300)
        if st.button("Analyze your query"):
            if legal_query is not None:
                context_docs = retrive_context(legal_query)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2
                )
                prompt_template = PromptTemplate(
                input_variables=["legal_query", "context_docs"],
                template="""
                You are a Lawyer or legal consultant. Analyze the document below against the legal query.

                Legal Query:
                {legal_query}

                Document Context:
                {context_docs}

                Provide a structured analysis of how well the document matches the legal query.                 
                """
                )

                chain = (
                    RunnableMap({
                        "legal_query": lambda x: x["legal_query"],
                        "context_docs": lambda x: x["context_docs"]
                    })
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                analysis = chain.invoke({
                    "legal_query": legal_query,
                    "context_docs": context_docs
                })

                st.header("AI Analysis")
                st.markdown(analysis)
            else:
                st.error("Please enter a legal query.")     
            
    with col2:
        st.header("Upload Legal Document")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf"]) 
        if st.button("Store Document to vector store"):
            if uploaded_file is not None:
                text = get_text_from_pdf(uploaded_file)
                store_vector_from_text(text, uploaded_file.name, doc_id=os.path.splitext(uploaded_file.name)[0])
                st.success("Document stored successfully.")
            else:
                st.error("Please upload a document.")
         
if __name__ == "__main__":
    main()