import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated per warning
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Path to your FAISS vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache the vector store loading
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Return a custom prompt template
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load a small LLM from Hugging Face
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-small",
        tokenizer="microsoft/DialoGPT-small",
        max_length=2000,       # ✅ Reduced from 512
        temperature=0.6,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Streamlit chatbot UI and logic
def main():
    st.title("🩺 Medical Chatbot")
    st.write("Ask any medical-related question based on our uploaded knowledge base.")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your Message here...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            llm = load_llm()

            # Custom QA prompt
            custom_prompt_template = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, say "I don't know." Do not make up information.
            Don't use any information outside the context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """
            custom_prompt = set_custom_prompt(custom_prompt_template)

            # Create Retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # ✅ Faster
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt}
            )

            # Run QA chain
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response.get("source_documents", [])

            # Show assistant's answer
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

            # Optional: Show sources in expandable section
            if source_documents:
                with st.expander("📚 Sources"):
                    for doc in source_documents:
                        st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"Error: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
