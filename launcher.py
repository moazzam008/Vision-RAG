import subprocess
import webbrowser
import time
import os
import sys
import urllib.request


def wait_for_backend(url, timeout=120):
    """Poll the backend until it responds or timeout is reached."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    print("=" * 60)
    print("  Visual Document Intelligence Pipeline")
    print("=" * 60)

    # Determine the base directory
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        # Check if the exe is in the parent folder
        if os.path.exists(os.path.join(exe_dir, "multimodal_rag", "venv")):
            MULTIMODAL_DIR = os.path.join(exe_dir, "multimodal_rag")
        else:
            MULTIMODAL_DIR = exe_dir
    else:
        MULTIMODAL_DIR = os.path.dirname(os.path.abspath(__file__))

    VENV_UVICORN = os.path.join(MULTIMODAL_DIR, "venv", "Scripts", "uvicorn.exe")
    VENV_STREAMLIT = os.path.join(MULTIMODAL_DIR, "venv", "Scripts", "streamlit.exe")

    if not os.path.exists(VENV_UVICORN):
        print(f"Error: Virtual environment not found in {MULTIMODAL_DIR}. Please run setup first.")
        input("Press Enter to exit...")
        return

    # Start FastAPI backend
    print("[1/3] Starting backend API on port 8000...")
    fastapi_process = subprocess.Popen(
        [VENV_UVICORN, "api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=MULTIMODAL_DIR
    )

    # Wait for backend to be ready BEFORE starting the frontend
    print("[2/3] Waiting for backend to be ready (this may take 15-30s)...")
    if wait_for_backend("http://127.0.0.1:8000/", timeout=120):
        print("       Backend is READY!")
    else:
        print("       WARNING: Backend did not respond in time. Continuing anyway...")

    # Start Streamlit frontend
    print("[3/3] Starting frontend UI on port 8501...")
    streamlit_process = subprocess.Popen(
        [VENV_STREAMLIT, "run", "frontend.py", "--server.headless=true"],
        cwd=MULTIMODAL_DIR
    )

    time.sleep(3)
    print("\nOpening browser...")
    webbrowser.open("http://localhost:8501")

    try:
        print("\n" + "=" * 60)
        print("  Application is running.")
        print("  Close this window to shut down all services.")
        print("=" * 60)
        fastapi_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down servers...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        # Also kill any child processes
        try:
            fastapi_process.kill()
            streamlit_process.kill()
        except Exception:
            pass
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
