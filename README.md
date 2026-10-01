# Multimodal Visual Document Intelligence (Vision RAG)

![Architecture Badge](https://img.shields.io/badge/Architecture-RAG-blue)
![Model Badge](https://img.shields.io/badge/Vision_Model-Qwen2--VL-orange)
![Database Badge](https://img.shields.io/badge/Vector_DB-Qdrant-red)

A high-performance Multimodal Retrieval-Augmented Generation (RAG) pipeline designed to extract, vectorize, and semantically search complex visual documents, engineering blueprints, and technical diagrams natively on CPU.

## 🚀 Features

- **Multimodal Ingestion**: Process images (`.jpg`, `.png`) and text documents (`.pdf`, `.docx`) in a single unified pipeline.
- **State-of-the-Art Vision-Language Extraction**: Leverages the natively integrated `Qwen2-VL-2B-Instruct` model to accurately transcribe measurements, labels, and structural features from complex visual data.
- **CPU-Optimized Pipeline**: Includes automatic high-resolution image downscaling and greedy decoding (`num_beams=1`) to prevent CPU OOM locks and ensure blazing-fast local processing times.
- **Semantic Vector Search**: Generates dense embeddings via `all-MiniLM-L6-v2` and stores them in a highly scalable local `Qdrant` vector database for natural language retrieval.
- **Robust Local Execution**: Packaged with a custom Streamlit frontend and FastAPI backend, completely offline-capable, and easily launched via a single Windows Executable (`Start_Visual_RAG.exe`).

## 🛠️ Tech Stack

- **Backend API**: FastAPI, Uvicorn
- **Frontend UI**: Streamlit
- **Machine Learning**: PyTorch, Hugging Face `transformers`
- **Vision-Language Model**: `Qwen/Qwen2-VL-2B-Instruct`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: Qdrant (Local)
- **Document Parsing**: Microsoft MarkItDown (`pdfplumber`, `pypdfium2`)

## ⚙️ How it Works

1. **Ingestion**: Uploaded documents are intercepted by FastAPI.
2. **Parsing**: 
   - *Visuals*: Automatically resized to max 512x512, passed through Qwen2-VL to generate highly detailed descriptive metadata.
   - *Documents*: Parsed to markdown using Microsoft MarkItDown.
3. **Embedding**: The extracted descriptions and text are embedded into a 384-dimensional dense vector space.
4. **Indexing**: Vectors and metadata are upserted into the local Qdrant collection.
5. **Retrieval**: User queries are embedded and compared against the vector database using Cosine Similarity to return the most relevant documents.

## 🏃‍♂️ Running Locally

Simply run the bundled executable to spin up the entire pipeline:
```bash
Start_Visual_RAG.exe
```
This automatically launches the FastAPI backend on port `8000` and the Streamlit frontend on port `8501`.

*Note: The very first run requires an active internet connection to download and cache the 3GB Qwen2-VL model weights. Subsequent runs are entirely offline.*
