# RAG Application with Streamlit

This is a basic Retrieval-Augmented Generation (RAG) application built with Streamlit, Langchain, ChromaDB, and OpenAI. It allows users to upload a document (PDF or TXT), provide an OpenAI API key, and then ask questions about the content of the document.

## Features

- **Document Upload:** Upload PDF or TXT files.
- **API Key Input:** Securely enter your OpenAI API key.
- **RAG Powered Q&A:** Ask questions about your uploaded document.
- **Conversational Memory:** The application maintains a short conversation history.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rag_application
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Using the Application:**
    *   Open your web browser and navigate to the address provided by Streamlit (usually `http://localhost:8501`).
    *   Enter your OpenAI API Key in the designated input field.
    *   Upload your document (PDF or TXT file).
    *   Once the document is processed, you can start asking questions in the "Ask a question about your document:" field.

## File Structure

-   `app.py`: The main Streamlit application file containing the UI and RAG logic.
-   `requirements.txt`: Lists all the Python dependencies required to run the application.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests to improve the application.

## License

This project is open-source and available under the MIT License.
