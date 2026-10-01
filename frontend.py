import streamlit as st
import requests
import os
import time

API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 600  # 10 minutes for large images on CPU


def call_api(method, endpoint, retries=5, delay=3, **kwargs):
    """Call the backend API with automatic retries for connection errors."""
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    for attempt in range(retries):
        try:
            if method == "post":
                return requests.post(f"{API_URL}{endpoint}", **kwargs)
            else:
                return requests.get(f"{API_URL}{endpoint}", **kwargs)
        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
    return None


st.set_page_config(
    page_title="Visual Document Intelligence",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3264/3264284.png", width=60)
    st.title("Vision RAG")
    st.markdown("---")
    st.markdown(
        "**Multimodal Retrieval-Augmented Generation (RAG)** pipeline designed for complex visual documents, "
        "engineering blueprints, and technical diagrams."
    )
    st.markdown("---")

    # Backend health check
    try:
        health = requests.get(f"{API_URL}/", timeout=3)
        st.success("✅ Backend Connected")
    except Exception:
        st.error("❌ Backend Offline")
        st.caption("Close the app and restart Start_Visual_RAG.exe")

    st.markdown("---")
    st.caption("Powered by Qwen2-VL & Qdrant")
    st.caption("Developed for High-Performance Visual Data Extraction")

st.title("👁️ Multimodal Visual Document Intelligence")
st.markdown("Automate the extraction, embedding, and semantic search of complex visual data.")

tab1, tab2 = st.tabs(["📤 Ingest Documents", "🔍 Semantic Search"])

with tab1:
    st.header("Document Ingestion Engine")
    st.markdown("Upload visual documents to be analyzed by the Vision-Language Model and stored in the vector database.")

    uploaded_file = st.file_uploader(
        "Upload Document (JPG, PNG, PDF, DOCX)",
        type=["jpg", "jpeg", "png", "pdf", "docx"],
        help="Upload a blueprint, diagram, or text document."
    )

    if st.button("Process & Ingest", type="primary") and uploaded_file is not None:
        with st.status("Processing Document...", expanded=True) as status:
            st.write("Uploading file to server...")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            st.write("Running Qwen2-VL Vision-Language analysis...")
            st.write("⏳ This may take 1-3 minutes on CPU. Please be patient...")

            try:
                response = call_api("post", "/ingest", files=files)

                if response.status_code == 200:
                    status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                    st.success("Document successfully vectorized and indexed.")
                    with st.expander("View VLM Extracted Metadata"):
                        st.json(response.json()["data"])
                else:
                    status.update(label="❌ Ingestion Failed", state="error")
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                status.update(label="❌ Backend Unreachable", state="error")
                st.error(
                    "Could not connect to the backend API. "
                    "Please close the app and restart **Start_Visual_RAG.exe**."
                )
            except Exception as e:
                status.update(label="❌ Ingestion Failed", state="error")
                st.error(f"Unexpected error: {e}")

with tab2:
    st.header("Knowledge Retrieval")
    st.markdown("Use natural language to search across all indexed visual documents.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., 'Show me diagrams containing a 12V battery'")
    with col2:
        top_k = st.number_input("Top K Results", min_value=1, max_value=10, value=3)

    if st.button("Search Documents", type="primary") and query:
        with st.spinner("Querying vector database..."):
            try:
                payload = {"query": query, "top_k": top_k}
                response = call_api("post", "/ask", json=payload)

                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if not results:
                        st.info("No matching documents found in the database.")

                    for idx, res in enumerate(results):
                        st.markdown(f"### Result {idx+1} | Confidence: `{res['score']:.4f}`")
                        r_col1, r_col2 = st.columns([1, 2])

                        with r_col1:
                            if os.path.exists(res["image_path"]):
                                st.image(res["image_path"], use_container_width=True, caption=os.path.basename(res["image_path"]))
                            else:
                                st.warning("Image file not found locally.")

                        with r_col2:
                            st.markdown("**VLM Context / Extracted Description:**")
                            st.info(res["description"])

                        st.divider()
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend API. "
                    "Please close the app and restart **Start_Visual_RAG.exe**."
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")
