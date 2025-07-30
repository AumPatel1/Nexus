#!/usr/bin/env python3
"""
Setup Guide for Three-Agent Medical Interview System

This script provides step-by-step instructions for getting started.
"""

import os
import sys

def print_header():
    """Print welcome header."""
    print("🏥 Three-Agent Medical Interview System")
    print("="*60)
    print("🎯 Setup Guide & Getting Started")
    print("="*60)

def check_python():
    """Check Python version."""
    print("\n1️⃣ Python Version Check")
    print("-" * 30)
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Good!")
    else:
        print(f"⚠️  Python {version.major}.{version.minor}.{version.micro} - Recommend Python 3.8+")
    
def check_dependencies():
    """Check if dependencies are installed."""
    print("\n2️⃣ Dependencies Check")
    print("-" * 30)
    
    packages = [
        ("streamlit", "Web UI framework"),
        ("langchain", "LangChain core"),
        ("langgraph", "LangGraph state management"),
        ("openai", "OpenAI API client"),
        ("langchain_google_genai", "Google Gemini support (optional)")
    ]
    
    missing = []
    
    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print("pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies installed!")

def show_api_key_guide():
    """Show how to get API keys for supported providers."""
    print("\n3️⃣ AI Provider API Key Setup")
    print("-" * 30)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    
    if openai_key:
        print(f"✅ OpenAI API Key found: {openai_key[:8]}...{openai_key[-4:]}")
    else:
        print("❌ No OpenAI API Key found")
        
    if gemini_key:
        print(f"✅ Google Gemini API Key found: {gemini_key[:8]}...{gemini_key[-4:]}")
    else:
        print("❌ No Google Gemini API Key found")
    
    print("\n📋 Option A: OpenAI API Key")
    print("1. Go to: https://platform.openai.com/api-keys")
    print("2. Sign in to your OpenAI account (or create one)")
    print("3. Click 'Create new secret key'")
    print("4. Give it a name (e.g., 'Medical Interview System')")
    print("5. Copy the key (starts with 'sk-')")
    
    print("\n📋 Option B: Google Gemini API Key")
    print("1. Go to: https://aistudio.google.com/app/apikey")
    print("2. Sign in to your Google account")
    print("3. Click 'Create API key'")
    print("4. Copy the key")
    print("5. Note: Gemini often has free tier available!")
    
    print("\n⚙️ How to set your API Key:")
    print("\n🌐 Option 1: Use the Web UI (Easiest)")
    print("   - Start the web app: python run_ui.py")
    print("   - Select AI provider (OpenAI or Gemini)")
    print("   - Enter your API key in the sidebar")
    print("   - It will be used for that session only")
    
    print("\n🖥️ Option 2: Set Environment Variable")
    if os.name == 'nt':  # Windows
        print("   Windows Command Prompt:")
        print("   set OPENAI_API_KEY=your-openai-key")
        print("   set GOOGLE_API_KEY=your-gemini-key")
        print("   python main.py")
        print("\n   Windows PowerShell:")
        print("   $env:OPENAI_API_KEY='your-openai-key'")
        print("   $env:GOOGLE_API_KEY='your-gemini-key'")
        print("   python main.py")
    else:  # Unix/Linux/Mac
        print("   Terminal:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export GOOGLE_API_KEY='your-gemini-key'")
        print("   python main.py")
    
    print("\n🔒 Security Tips:")
    print("   - Never commit your API key to git")
    print("   - Don't share your key with others")
    print("   - Use environment variables or the web UI")
    print("   - Monitor usage at platform.openai.com")

def show_usage_guide():
    """Show how to use the system."""
    print("\n4️⃣ How to Use the System")
    print("-" * 30)
    
    print("\n🌐 Web UI (Recommended):")
    print("python run_ui.py")
    print("   ↳ Beautiful interface with real-time conversation")
    print("   ↳ Enter API key in sidebar")
    print("   ↳ Select patient persona")
    print("   ↳ Watch the three agents in action!")
    
    print("\n🖥️ Command Line:")
    print("python main.py")
    print("   ↳ Runs 4 patient scenarios automatically")
    print("   ↳ Requires API key in environment")
    print("   ↳ Great for batch testing")

def show_architecture_summary():
    """Show system architecture summary."""
    print("\n5️⃣ System Architecture")
    print("-" * 30)
    
    print("🤖 Three AI Agents Working Together:")
    print("   🩺 Interviewer Agent (LangGraph)")
    print("      ↳ Asks questions following medical protocol")
    print("      ↳ Routes conversation based on responses")
    print("      ↳ Tracks symptoms and risk levels")
    
    print("   👤 Patient Agent (LangChain)")
    print("      ↳ Simulates realistic patient responses")
    print("      ↳ Maintains consistent persona")
    print("      ↳ Natural conversation style")
    
    print("   ⚖️ Judge Agent (LangChain)")
    print("      ↳ Evaluates interview quality")
    print("      ↳ Monitors safety protocols")
    print("      ↳ Provides detailed scoring")

def show_next_steps():
    """Show what to do next."""
    print("\n6️⃣ Next Steps")
    print("-" * 30)
    
    print("🚀 Ready to start? Choose your path:")
    print("\n   🌐 For easy exploration:")
    print("      python run_ui.py")
    
    print("\n   🖥️ For command line:")
    print("      export OPENAI_API_KEY='your-key'")
    print("      python main.py")
    
    print("\n📚 Want to customize?")
    print("   - Edit patient personas in main.py")
    print("   - Modify questions in mini_module_a.json")
    print("   - Adjust agent behavior in agents.py")
    
    print("\n❓ Need help?")
    print("   - Check README.md for detailed docs")
    print("   - Look at the code comments")
    print("   - Architecture diagram in the web UI")

def main():
    """Main setup guide function."""
    print_header()
    check_python()
    check_dependencies()
    show_api_key_guide()
    show_usage_guide()
    show_architecture_summary()
    show_next_steps()
    
    print("\n" + "="*60)
    print("🎉 You're all set! Happy interviewing!")
    print("="*60)

if __name__ == "__main__":
    main() 