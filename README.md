# Aurora Health Chatbot

![Aurora Health Chatbot Logo](assets/logo.jpg)

## Overview

Aurora Health Chatbot is an AI-powered personal health assistant built with Streamlit, LangChain, OpenAI, and Pinecone. It allows users to upload health-related documents (PDFs and Excel files, such as lab results, medical records, or reports), indexes them into a vector database, and provides natural language responses to queries about the content. The chatbot uses Retrieval-Augmented Generation (RAG) to fetch relevant document chunks and generate accurate, context-aware answers.

Key technologies:
- **Frontend**: Streamlit for the interactive UI.
- **Backend**: LangChain for document processing, embeddings, and chaining; OpenAI for embeddings and LLM; Pinecone for vector storage.
- **Features**: Voice input/output, file uploads for dynamic database updates, structured responses with tables and emojis, and source document references.

This project is ideal for personal health management, allowing users to query their records conversationally (e.g., "What were my cholesterol levels in 2023?" or "Compare my blood pressure trends over time").

## Features

- **Document Upload and Indexing**: Upload PDFs or Excel files (e.g., lab reports) to add them to your personal health database. Files are processed, chunked, and stored in Pinecone.
- **Query Handling**: Ask questions via text input or voice (using speech-to-text).
- **RAG-Powered Responses**: Retrieves relevant document chunks and generates professional, empathetic responses with:
  - Tables for comparisons (e.g., test results over time).
  - Emojis for status indicators (✅ normal, ❌ high risk, ⚠ borderline).
  - Trends, summaries, and doctor consultation reminders.
- **Voice Response**: Optional text-to-speech output with customizable voices.
- **Chat History**: Maintains conversation context for follow-up questions.
- **Source Transparency**: Displays metadata from retrieved documents (e.g., file name, page/sheet) in an expander.
- **Reset Functionality**: Clear chat history with a button.
- **Error Handling**: Graceful handling for missing data or API issues.

## Prerequisites

- Python 3.10+ (tested on 3.12).
- API Keys:
  - OpenAI API Key (for embeddings and LLM).
  - Pinecone API Key (for vector database).
- A "data" folder with initial PDF/Excel files (optional; you can upload files via the app).
- Assets: Place logo images in an `assets/` folder (e.g., `logo.jpg`, `logo-.png`, `user.png`, `assistant.png`).


Run:
```
pip install -r requirements.txt
```

Note: Some libraries (e.g., `unstructured`) may require additional system dependencies like `poppler` for PDFs or `tesseract` for OCR—install them via your package manager if errors occur.



### Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

- **Pinecone Index**: The default index name is `health-docs`. It's created automatically if it doesn't exist (dimension: 1536, metric: cosine, cloud: AWS us-east-1). Change the region in `vector_db_pipeline.py` if needed.
- **OpenAI Models**: Embeddings use `text-embedding-3-small` (1536 dimensions). LLM uses `gpt-5` (update to a valid model like `gpt-4o` if unavailable).
- **Chunking**: Documents are split into chunks of 800 characters with 200 overlap (configurable in `vector_db_pipeline.py`).
- **Voices**: TTS uses Edge-TTS with predefined voices (William, James, Jenny, US Guy). Enable in the sidebar toggle.

## Usage

### Step 1: Index Initial Documents (Optional)
If you have files in the `data/` folder, run the pipeline to index them into Pinecone:
```
python vector_db_pipeline.py
```
This loads, processes, chunks, and embeds documents. It prints a summary (e.g., files processed, chunks created).

### Step 2: Run the Chatbot
Launch the Streamlit app:
```
streamlit run app.py
```
- Open in your browser (default: http://localhost:8501).
- **Sidebar**:
  - Upload new PDFs/Excel files—they'll be added to the database dynamically.
  - Toggle "Enable Voice Response" and select a voice.
- **Main Interface**:
  - Chat history starts with a welcome message.
  - Input queries via text box or voice button.
  - Responses appear with structured formatting.
  - Use the "Reset" button to clear history.
- **Example Queries**:
  - "Summarize my latest lab results."
  - "What is my blood sugar trend over the last year?"
  - "Compare my cholesterol levels from 2022 and 2023."

### Adding New Files
- Upload via the sidebar file uploader. The app processes and adds them to Pinecone automatically.
- No need to restart—the vector store updates in real-time.

### Testing Search
In `vector_db_pipeline.py`, after running the pipeline, you can test similarity search:
```python
results = pipeline.search_similarity("query about health data", k=5)
print(results)
```

## How It Works

1. **Document Processing (`vector_db_pipeline.py`)**:
   - Loads PDFs (page-by-page) and Excel (sheet-by-sheet) with metadata (e.g., file hash, dates, sizes).
   - Splits into chunks using `RecursiveCharacterTextSplitter`.
   - Embeds with OpenAI and stores in Pinecone.
   - Handles duplicates via file hashes.

2. **Chatbot Logic (`app.py`)**:
   - Retrieves from Pinecone using similarity search (top 6 chunks).
   - Contextualizes questions with chat history.
   - Generates responses via LangChain chain with a custom prompt (professional, health-focused).
   - Formats outputs with tables, emojis, and sources.
   - Supports speech-to-text input and text-to-speech output.

3. **Error Handling**:
   - If no relevant data: "I'm sorry, I don't have that information..."
   - API errors: Displayed in the UI.

## Troubleshooting

- **API Key Issues**: Ensure `.env` is loaded and keys are valid.
- **Pinecone Connection**: Check index name and region; delete/recreate index via Pinecone dashboard if needed.
- **Voice Features**: Requires microphone access; TTS may fail on non-supported platforms.
- **Large Files**: Chunking handles them, but very large Excels may need memory adjustments.
- **Model Availability**: If `gpt-5` is invalid, change to `gpt-4` or `gpt-3.5-turbo` in `app.py`.
- Logs: Check console for detailed errors (logging level: INFO).
