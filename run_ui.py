#!/usr/bin/env python3
"""
Launch script for the Three-Agent Medical Interview System Web UI

This script will start the Streamlit web interface.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import langchain
        import langgraph
        import openai
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n📦 Please install required packages:")
        print("pip install -r requirements.txt")
        return False

def main():
    print("🏥 Three-Agent Medical Interview System")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ Dependencies found!")
    print("\n🚀 Starting Streamlit web interface...")
    print("\n📋 Instructions:")
    print("1. The web app will open in your browser")
    print("2. Enter your OpenAI API key in the sidebar")
    print("3. Select a patient persona")
    print("4. Click 'Start Interview' to begin")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("="*50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 