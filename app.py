import os
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with PDF")
st.write("Upload a PDF and ask questions about its content.")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is missing.")
    st.info(
        "Create a .env file in the same folder as app.py and add:\n\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )
    st.stop()


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )


@st.cache_resource
def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    return FAISS.from_documents(chunks, embeddings)


def answer_question(question, vector_store):
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = """
You are a helpful assistant answering questions from a PDF.

Answer using only the information in the context.
If the answer is not present, say:
"I could not find the answer in the uploaded PDF."

Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = get_llm().invoke(final_prompt)

    # IMPORTANT: return only text, not type/text/extras metadata.
    return response.text


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    temp_path = "uploaded_document.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    try:
        vector_store = create_vector_store(temp_path)
        st.success("PDF processed successfully. You can now ask questions.")

        question = st.text_input("Ask a question about your PDF:")

        if question:
            with st.spinner("Generating answer..."):
                answer = answer_question(question, vector_store)

            st.subheader("Answer")
            st.write(answer)

    except Exception as e:
        st.error("An error occurred while processing the PDF.")
        st.exception(e)