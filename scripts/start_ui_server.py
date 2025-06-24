#!/usr/bin/env python3
"""
Script to start the Data Quality Summarizer UI server.
Runs both the FastAPI backend and serves the React frontend.
"""

import sys
import subprocess
import os
import time
import signal
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_quality_summarizer.ui.backend_integration import run_server

def check_react_build():
    """Check if React build exists, if not provide instructions."""
    build_dir = Path(__file__).parent.parent / "dist" / "ui"
    
    if not build_dir.exists():
        print("\n🔧 React UI not built yet. Building now...")
        print("This may take a few minutes the first time...\n")
        
        # Change to project root
        project_root = Path(__file__).parent.parent
        
        try:
            # Install npm dependencies if needed
            if not (project_root / "node_modules").exists():
                print("📦 Installing npm dependencies...")
                subprocess.run(["npm", "install"], cwd=project_root, check=True)
            
            # Build React app
            print("🔨 Building React application...")
            subprocess.run(["npm", "run", "build"], cwd=project_root, check=True)
            
            print("✅ React build completed!\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building React app: {e}")
            print("\nTo build manually, run:")
            print("  npm install")
            print("  npm run build")
            return False
        except FileNotFoundError:
            print("❌ npm not found. Please install Node.js and npm first.")
            print("Visit: https://nodejs.org/")
            return False
    
    return True

def start_development_servers():
    """Start both React dev server and FastAPI backend in development mode."""
    print("🚀 Starting development servers...")
    print("  - FastAPI backend: http://localhost:8000")
    print("  - React frontend: http://localhost:3000")
    print("\nPress Ctrl+C to stop both servers\n")
    
    project_root = Path(__file__).parent.parent
    
    # Start React dev server
    react_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give React a moment to start
    time.sleep(2)
    
    try:
        # Start FastAPI server
        run_server(host="127.0.0.1", port=8000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        react_process.terminate()
        react_process.wait()
        print("✅ Servers stopped")

def start_production_server():
    """Start production server with built React app."""
    if not check_react_build():
        return
    
    print("🚀 Starting production server...")
    print("  Server: http://localhost:8000")
    print("  UI: http://localhost:8000")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        run_server(host="127.0.0.1", port=8000, debug=False)
    except KeyboardInterrupt:
        print("\n✅ Server stopped")

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        start_development_servers()
    else:
        start_production_server()

if __name__ == "__main__":
    main()