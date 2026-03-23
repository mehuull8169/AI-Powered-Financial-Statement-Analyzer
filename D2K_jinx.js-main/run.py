import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def run_command(command):
    """Run a command in a new process"""
    process = subprocess.Popen(command, shell=True)
    return process

def setup_environment():
    """Setup the environment for the application"""
    # Create temp_uploads directory if it doesn't exist
    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Check if .env file exists and has API key
    if not os.path.exists(".env"):
        print("Error: .env file not found!")
        return False
    
    with open(".env", "r") as f:
        if "GOOGLE_API_KEY" not in f.read():
            print("Error: GOOGLE_API_KEY not found in .env file!")
            return False
    
    return True

def main():
    """Main function to run the application"""
    if not setup_environment():
        return
    
    print("Starting FastAPI backend...")
    backend = run_command("start cmd /k python -m uvicorn backend:app --reload")
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(3)
    
    print("Starting Streamlit frontend...")
    frontend = run_command("start cmd /k streamlit run app.py")
        
    print("\n===== Financial Statement Analysis Platform Started =====")
    print("Backend API: http://localhost:8000")
    print("Frontend UI: http://localhost:8501")
    print("\nPress Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        # The processes will be terminated when their command windows are closed

if __name__ == "__main__":
    main()
