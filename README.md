Legal document review application with LangChain

Overview
This is a Legal document review application developed using LangChain that lawyers & legal team in efficiently screening legal documents.The application pulls text from uploaded documents, analyzes them based on an AI model, and gives a detailed report.

Scenario
Legal professionals frequently deal with large volumes of complex documents such as contracts, NDAs, and terms of service. Reviewing these documents manually is  time intensive and prone to human error. A solution that leverages AI to help users understand, summarize, and query these legal documents can significantly improve
productivity and reduce legal risk.This assignment tasks you with building a web application that enables users to upload legal documents and receive intelligent, context-aware assistance through modern language models.

Problem Statement
Lawyers and legal teams require an efficient tool to quickly review and understand the contents of legal documents. Manually identifying key clauses or summarizing lengthy
texts is inefficient and costly. There is a need for a user-friendly application that can extract text from legal PDFs, summarize key information, and answer document-specific questions using Retrieval-Augmented Generation (RAG) and large language models.


Approach:-
Building a Legal document review Application using LangChain and Streamlit, where users can:
- Input question in a text area.
- Upload a legal document in PDF format & store into FAISS store.
- Analyze the document using Google’s Gemini 2.0 Flash model to get a detaied analysis.


The application uses LangChain document loaders (PyPDFLoader) to extract text from legal documents, vectorise the text, store it into FAISS store and LangChain’s LLMChain with a custom prompt to generate a structured analysis. Streamlit provides a user-friendly interface with custom styling for better readability.

Project Structure
- app.py: Main application code for the legal document review Application.
- requirements.txt: List of dependencies required to run the application.
- .env: Environment file to store the Google API key (GOOGLE_API_KEY).
- venv/: Virtual environment directory (e.g., env for storing installed packages).

Setup Instructions:-
Create a Virtual Environment:
- Navigate to the project directory.
- Run: python -m venv venv (or env if preferred).
- Activate the virtual environment:
- Windows: venv\Scripts\activate
- macOS/Linux: source venv/bin/activate

Install Dependencies:
- Ensure requirements.txt is in the project directory.
- Run: pip install -r requirements.txt

Set Up the API Key:
- Create a .env file in the project directory.
- Add your Google API key: GOOGLE_API_KEY=your_api_key_here
- Ensure the .env file is loaded using python-dotenv (already included in app.py).

Run the Application Locally:
- Run: streamlit run app.py
- Open the provided URL (e.g., http://localhost:8501) in your browser to access the app.

Deploying on Streamlit Cloud:-

Prepare Your Project:
- Ensure app.py, requirements.txt, and .env are in the project directory.
- Create a Streamlit Cloud account at https://streamlit.io/cloud.

Upload to Streamlit Cloud:
- Log in to Streamlit Cloud.
- Create a new app and connect it to your project directory (e.g., upload the files manually or link to a cloud storage service).
- Specify app.py as the main script.

Configure Environment Variables:
- In Streamlit Cloud, go to your app’s settings.
- Add the GOOGLE_API_KEY as a secret environment variable (do not include .env in the uploaded files for security).

Deploy the App:
- vClick “Deploy” in Streamlit Cloud.
- Once deployed, access the app via the provided URL (e.g., https://your-app-name.streamlit.app).

Test the Deployed App:
- Upload legal documents and store it into FAISS store
- Input legal queries and analyze it.
- Verify that the analysis report is generated.

Requirements
The requirements.txt file includes all necessary dependencies.

Usage
- Run the app locally or access the deployed version on Streamlit Cloud.
- Enter queries in the text area (e.g., Deed, agreement).
- Upload a document in PDF format.
- Click “Analyze your query” to generate the AI-driven analysis.

## 🚀 Features

- Upload document in PDF formats
- Extract and store file embedding into FAISS store
- Run similarity search on FAISS store using a query
- Pass relavant context and query to gemini LLM to analyse it
- Dislay analysis on a UI
- View structured AI-generated feedback


---

## 📦 Tech Stack

- **LangChain** for building chains, embeddings, and document processing
- **LangChain Expression Language (LCEL)** for modular pipeline workflows
- **Streamlit** for the frontend web interface
- **Google Generative AI** (Gemini & Embeddings) for LLM and vector representations
- **FAISS store** as a persistent vector store
- **dotenv** for API key and environment config

---

## 🛠️ Setup Instructions
````

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**
   Create a `.env` file in the project root and add:

   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📄 File Structure

```plaintext
├── app.py                  # Main Streamlit app
├── FAISS_store/            # Folder to store vector DB files
├── .env                    # Contains API key (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📚 LangChain Concepts Used

* ✅ **Components & Modules**: PromptTemplate, LLM, Output Parsers
* 📄 **Document Loaders**: PDF, DOCX, TXT via LangChain community
* ✂️ **Text Splitting**: RecursiveCharacterTextSplitter
* 🧠 **Embeddings**: GoogleGenerativeAIEmbeddings
* 🗃️ **Vector DB**: FAISS for persistent storage
* 🧩 **LCEL**: RunnableMap, pipes (`|`), and chain composition
* 🧪 **Chains**: Custom chain for job/resume comparison
* 📤 **Deployment**: Streamlit as the UI layer

---

## 📈 Example Output

```
Structured Analysis:
- Strengths: Relevant details etc.
- Weaknesses: Lacks X, missing Y...

```

---

## 🧑‍💼 Ideal For

* Lawyer and legal consultant
* Legal document screening automation tools
* Educational and project demos for LangChain and LCEL