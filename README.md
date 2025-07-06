# Medical_chatbot_LLM
Hey there! 👋
This project is an AI-powered **medical chatbot** that uses **large language models** (LLMs) from **Hugging Face**, runs logic and memory through **LangChain**, and provides a clean, interactive frontend using **Streamlit**. The goal is to offer helpful, conversational medical information — not as a doctor, but like a smart assistant that can understand your questions and provide informed responses.

---

##  What This Chatbot Can Do

*  Understand and answer general health-related questions.
*  Pull context-aware responses using LangChain.
*  Provide a smooth chat experience via a Streamlit UI.
*  Ready to be expanded — think symptom checkers, report uploads, or doctor-patient messaging.

>  Example questions:
>
> * *“What are the early signs of diabetes?”*
> * *“How do I treat a fever at home?”*
> * *“What does an MRI scan show?”*

---

##  Tech Used

* **Hugging Face Transformers** – for powerful LLMs
* **LangChain** – to manage the chat flow and memory
* **Streamlit** – for the web-based UI
* **Python** – because of course

---

## Getting Started

Here’s how to run this on your machine:

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Add your Hugging Face API token

Create a `.env` file and paste your token inside:

```env
HF_TOKEN=your_huggingface_token_here
```

### 4. Run the app

```bash
streamlit run connect_memory_with_llm.py
```

Once it launches, you can start chatting directly from your browser.


## 💬 Why This Project?

I wanted to build something useful at the intersection of AI and healthcare — something that could help users better understand their symptoms or get health info, especially when a doctor isn't immediately available. This chatbot is still in its early stages, but it's already able to hold decent conversations and give helpful medical guidance (again, **not** a replacement for real medical advice).

---

## ⚠️ Disclaimer

> **This is not a real doctor.**
> The chatbot is designed for educational and informational purposes only. For medical emergencies or professional diagnosis, always consult a licensed healthcare provider.

---

## 🛠️ What’s Next?

Here are some cool ideas I’m working on (or plan to):

* [ ] Upload medical reports (PDF, X-ray, etc.)
* [ ] Match symptoms to possible conditions
* [ ] Add doctor dashboard and chat feature
* [ ] Enable medical document search with Retrieval-Augmented Generation (RAG)



## 🙏 Thanks

Shout out to the teams at:

* [Hugging Face](https://huggingface.co/)
* [LangChain](https://www.langchain.com/)
* [Streamlit](https://streamlit.io/)


