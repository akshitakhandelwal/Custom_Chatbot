# 🧠 Multi-Document QA Chatbot with Groq + Streamlit

This is a custom Retrieval-Augmented Generation (RAG) app that allows users to input multiple article URLs and ask intelligent questions. It uses:

-  Groq's LLM (LLaMA 3 via ChatGroq)
-  LangChain for sub-question decomposition
-  Vector search using ChromaDB
-  Smart document parsing and retrieval
-  Built-in multi-hop reasoning
-  Streamlit UI

## 🔧 Features

- Input up to 3 article URLs
- Automatically splits and embeds content
- Handles both simple and complex multi-hop questions
- Generates a final answer from sub-answers
- Easy-to-use Streamlit interface
