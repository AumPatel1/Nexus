#!/usr/bin/env python3
"""
Quick installer for Google Gemini support

Run this to add Gemini to your three-agent medical interview system.
"""

import subprocess
import sys

def install_gemini():
    """Install Google Gemini support for langchain."""
    print("🚀 Installing Google Gemini support...")
    print("=" * 50)
    
    try:
        # Install the package
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "langchain-google-genai"
        ])
        
        print("\n✅ SUCCESS! Google Gemini support installed!")
        print("\n📋 Next steps:")
        print("1. Get your API key from: https://aistudio.google.com/app/apikey")
        print("2. Restart your web app: python run_ui.py")
        print("3. Select 'Google Gemini' from the AI Provider dropdown")
        print("4. Enter your Gemini API key")
        print("\n🎉 You're ready to use Gemini!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\n🔧 Try manually:")
        print("pip install langchain-google-genai")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    install_gemini() 