
import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not set. Please set it in environment variables or Streamlit secrets.")
    st.stop()


llm = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)


st.set_page_config(page_title="Multi-Hop QA with Groq", layout="centered")
st.title("🧠 Multi-Document QA with Smart Decomposition")


url1 = st.text_input("🔗 URL 1")
url2 = st.text_input("🔗 URL 2")
url3 = st.text_input("🔗 URL 3")


if st.button("📚 Process Articles"):
    with st.spinner("Processing articles..."):
        urls = [url.strip() for url in [url1, url2, url3] if url.strip()]
        loader = WebBaseLoader(urls)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)


        st.session_state.vectorstore = vectorstore
        st.session_state.docs = docs
        st.success("✅ Articles processed successfully!")


subq_prompt = PromptTemplate.from_template(
    """
You are a smart assistant. Only decompose complex, multi-step questions into smaller sub-questions.
If the question is simple, return just one sub-question that reflects the original query.

User question: {question}

Sub-questions:
1.
"""
)
subq_chain = LLMChain(llm=llm, prompt=subq_prompt)

if "vectorstore" in st.session_state:
    question = st.text_input("💬 Ask a question")

    if st.button("🎯 Get Answer"):
        with st.spinner("Thinking..."):
            try:
                raw_subqs = subq_chain.run(question)
                subqs = [q.strip() for q in raw_subqs.split("\n") if q.strip()]

                if not subqs:
                    st.error("⚠️ No valid sub-questions were generated. Try rephrasing the question.")
                    st.stop()

                st.markdown("### 🧩 Sub-questions:")
                for i, sq in enumerate(subqs):
                    st.write(f"{i+1}. {sq}")

                retriever = st.session_state.vectorstore.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

                if len(subqs) == 1:
                    st.write("🧠 Simple question detected. Answering directly.")
                    final_answer = qa_chain.run(subqs[0])
                    st.success(final_answer)
                else:
                    sub_answers = [(subq, qa_chain.run(subq)) for subq in subqs]
                    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in sub_answers])

                    synthesis_prompt = PromptTemplate.from_template(
                        """
You are a helpful assistant. Use the following sub-questions and their answers to create a coherent final answer.

{context}

Final Answer:
"""
                    )
                    synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)
                    final_answer = synthesis_chain.run(context=context)
                    st.markdown("### 📘 Final Answer:")
                    st.success(final_answer)

            except Exception as e:
                st.error(f"🚨 An error occurred: {e}")
