import os
from langchain_huggingface.llms import HuggingFaceEndpoint  # updated!
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load token and repo ID
HF_TOKEN =os.getenv("HF_TOKEN")
huggingface_repo_id = "tiiuae/falcon-7b-instruct"


# Load the LLM
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.6,
        max_new_tokens=512,
       task="text-generation",
    )


# Prompt template
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't use any information outside the context.
Context: {context}
Question: {question}
Start the answer directly, NO small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# Ask user
user_query = input("Enter your question: ")
response = qa_chain.invoke({"query": user_query})
print("Answer:", response['result'])
print("Source Documents:", response['source_documents'])
