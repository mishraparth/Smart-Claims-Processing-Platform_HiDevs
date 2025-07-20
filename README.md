# Smart Claims Processing Platform

This project is automatically generated.

## Installation

```sh
pip install -r requirements.txt
```

-----

# 📄 Smart Claims Processing Platform

A powerful AI-driven application designed to streamline insurance claims processing using Retrieval-Augmented Generation (RAG). This tool analyzes policy documents and answers natural language questions about claim coverage, helping agents make faster, more accurate decisions.

-----

## ✨ Features

  * **Intelligent Document Analysis:** Automatically loads and processes text from PDF insurance policy documents.
  * **AI-Powered Q\&A:** Ask complex questions about policy coverage (e.g., "Is water damage from a burst pipe covered?") and get instant, context-aware answers.
  * **Policy Compliance Check:** The AI cites the exact sections of the policy document it used to generate its answer, ensuring transparency and compliance.
  * **Interactive UI:** A simple and intuitive web interface built with Streamlit.

-----

## 🛠️ Tech Stack

  * **Backend:** Python
  * **AI/ML Framework:** LangChain 🦜🔗
  * **LLM Provider:** Groq with Llama 3
  * **Embeddings:** Sentence-Transformers (from Hugging Face)
  * **Vector Database:** ChromaDB
  * **Frontend:** Streamlit 🎈

-----

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/mishraparth/Smart-Claims-Processing-Platform_HiDevs.git
    cd Smart-Claims-Processing-Platform_HiDevs
    ```

2.  **Create an environment file:**
    Create a file named `.env` in the root of the project and add your Groq API key:

    ```
    GROQ_API_KEY="YOUR_API_KEY_HERE"
    ```

3.  **Install dependencies:**
    Make sure you have Python 3.8+ installed. Then, run:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Policy Documents:**
    Place your insurance policy PDF files inside the `data/` directory. A `sample_policy.pdf` is included to get you started.

-----

## ▶️ How to Use

1.  **Run the application:**
    Open your terminal in the project directory and run the following command:

    ```bash
    streamlit run app.py
    ```

2.  **Process Documents:**
    Your browser will open with the application. First, click the **"Process Policy Documents"** button. This will read, chunk, and store the policy information in the vector database.

3.  **Ask Questions:**
    Once processing is complete, type your question into the text box and press Enter. The AI will analyze the query, retrieve relevant policy information, and generate a response.
