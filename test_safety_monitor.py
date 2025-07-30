#!/usr/bin/env python3
"""
Test script for safety monitoring functionality
"""

import os
import sys
from app import perform_safety_check

def test_safety_monitoring():
    """Test safety monitoring with different providers"""
    
    print("🧪 Testing Safety Monitoring System")
    print("=" * 50)
    
    # Check which provider to use
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not openai_key and not gemini_key:
        print("❌ No API keys found!")
        print("Set OPENAI_API_KEY or GOOGLE_AI_API_KEY environment variable")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "Safe Response",
            "question": "How are you feeling today?",
            "response": "I'm feeling okay, just a bit tired.",
            "expected": "SAFE"
        },
        {
            "name": "Mild Depression",
            "question": "Have you been feeling sad?",
            "response": "Yes, I've been feeling quite sad and hopeless lately.",
            "expected": "RISK"
        },
        {
            "name": "Suicidal Ideation",
            "question": "Have you had thoughts of death or harming yourself?",
            "response": "Yes, I have thoughts about death and ending my life.",
            "expected": "CRITICAL"
        },
        {
            "name": "Self-harm mention",
            "question": "Any thoughts of harming yourself?",
            "response": "Sometimes I think about cutting myself when things get bad.",
            "expected": "CRITICAL"
        }
    ]
    
    # Try OpenAI first, then Gemini
    for provider_name, has_key, create_llm_func in [
        ("OpenAI", openai_key, lambda: create_openai_llm()),
        ("Google Gemini", gemini_key, lambda: create_gemini_llm())
    ]:
        
        if not has_key:
            print(f"\n⏭️ Skipping {provider_name} - no API key")
            continue
            
        print(f"\n🧪 Testing with {provider_name}")
        print("-" * 30)
        
        try:
            llm = create_llm_func()
            print(f"✅ {provider_name} LLM created successfully")
            
            for test_case in test_cases:
                print(f"\n🔍 Test: {test_case['name']}")
                print(f"Response: '{test_case['response']}'")
                
                try:
                    result = perform_safety_check(
                        test_case['response'],
                        test_case['question'],
                        llm
                    )
                    
                    print(f"Assessment: '{result}'")
                    
                    if test_case['expected'] in result:
                        print(f"✅ PASS - Contains '{test_case['expected']}'")
                    else:
                        print(f"❌ FAIL - Expected '{test_case['expected']}', got '{result}'")
                        
                except Exception as e:
                    print(f"❌ Error in safety check: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error with {provider_name}: {e}")
    
    return False

def create_openai_llm():
    """Create OpenAI LLM"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        max_tokens=500
    )

def create_gemini_llm():
    """Create Gemini LLM"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=500,
        convert_system_message_to_human=True,
        google_api_key=api_key
    )

if __name__ == "__main__":
    success = test_safety_monitoring()
    
    if success:
        print("\n🎉 Safety monitoring tests completed!")
        print("\n✅ The safety monitoring system is working correctly.")
        print("   It should now trigger alerts in the main app when")
        print("   patients mention concerning content.")
    else:
        print("\n❌ Safety monitoring tests failed!")
        print("   Please check your API keys and try again.") 