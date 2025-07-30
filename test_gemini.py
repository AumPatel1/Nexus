#!/usr/bin/env python3
"""
Simple test script to verify Gemini API connection
"""

import os
import sys

def test_gemini_connection():
    """Test Gemini API connection with user's API key"""
    
    # Check if langchain-google-genai is installed
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ langchain-google-genai is installed")
    except ImportError:
        print("❌ langchain-google-genai is not installed")
        print("Install with: pip install langchain-google-genai")
        return False
    
    # Get API key from user
    api_key = input("Enter your Google AI API key: ").strip()
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # Set environment variables
    os.environ["GOOGLE_AI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Test different models
    models_to_test = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.3,
                max_output_tokens=100,
                convert_system_message_to_human=True,
                google_api_key=api_key
            )
            
            response = llm.invoke("Hello, respond with just 'Hello back!'")
            print(f"✅ {model} works! Response: {response.content}")
            return True
            
        except Exception as e:
            print(f"❌ {model} failed: {e}")
            if "authentication" in str(e).lower():
                print("🔑 Authentication error - check your API key")
            elif "not found" in str(e).lower():
                print("📝 Model not available, trying next...")
                continue
            else:
                print(f"🔍 Error details: {e}")
    
    print("\n❌ All models failed")
    return False

if __name__ == "__main__":
    print("🏥 Gemini API Connection Test")
    print("=" * 40)
    
    success = test_gemini_connection()
    
    if success:
        print("\n🎉 Gemini is working! You can use it in the main app.")
    else:
        print("\n💡 Troubleshooting tips:")
        print("1. Get API key from: https://aistudio.google.com/app/apikey")
        print("2. Make sure the key is valid and has proper permissions")
        print("3. Try a different model if one doesn't work")
        print("4. Check your internet connection") 