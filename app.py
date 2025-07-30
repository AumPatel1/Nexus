#!/usr/bin/env python3
"""
Streamlit Web UI for Three-Agent Medical Interview System

This creates a beautiful web interface showing:
- System architecture diagram
- Google Gemini API key configuration
- Real-time conversation between Interview and Patient agents
- Judge agent evaluation
"""

import streamlit as st
import os
import json
import uuid
import logging
from typing import Dict, Any, List
from datetime import datetime
import time

# Set up simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="🏥 Three-Agent Medical Interview System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for message styling only
st.markdown("""
<style>
    .interviewer-msg {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        color: #1565c0;
    }
    .patient-msg {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        color: #2e7d32;
    }
    .judge-msg {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        color: #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

# Import our modules (with error handling for missing dependencies)
try:
    from langgraph.graph import StateGraph
    from utils import (
        AgentState, create_initial_state, add_to_transcript, 
        is_interview_complete, RiskLevel, InterviewStatus
    )
    from agents import (
        get_question_node, parse_response_node, route_logic_node,
        create_patient_agent, create_judge_agent, create_safety_monitor_agent,
        generate_diagnosis, load_question_data
    )
    from main import build_interviewer_graph, load_patient_personas
    
    # Import Gemini support
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        
    DEPENDENCIES_AVAILABLE = GEMINI_AVAILABLE
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    GEMINI_AVAILABLE = False
    IMPORT_ERROR = str(e)

def show_architecture():
    """Display the system architecture - simplified."""
    st.markdown("## 🏥 Three-Agent Medical Interview System")
    
    # Simple one-line description
    st.info("🩺 **AI Interviewer** conducts depression screening with 👤 **Simulated Patient** while ⚖️ **Safety Judge** monitors")
    
    st.markdown("---")

def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("🔧 Configuration")
    
    # Gemini API Key Configuration
    st.sidebar.markdown("### 🔑 Google Gemini API Key")
    api_key = st.sidebar.text_input(
        "Enter your Google AI API Key:",
        type="password",
        help="Get your API key from https://aistudio.google.com/app/apikey"
    )
    
    if api_key:
        # Set both possible environment variable names
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_AI_API_KEY"] = api_key
        st.sidebar.success("✅ Gemini API Key configured!")
    else:
        st.sidebar.warning("⚠️ Please enter your Google AI API Key to continue")
    
    st.sidebar.markdown("---")
    
    # Model Configuration
    st.sidebar.markdown("### ⚙️ Model Settings")
    
    # Use Gemini model only
    model = st.sidebar.selectbox(
        "Gemini Model:",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        index=0
    )
    
    # Fixed temperature for consistency
    temperature = 0.3
    
    # Debug info for API configuration
    if api_key:
        if st.sidebar.checkbox("🔍 Show Debug Info", key="provider_debug"):
            st.sidebar.info(f"API key starts with: {api_key[:8]}...")
            st.sidebar.info(f"Using model: {model}")
            st.sidebar.info(f"Temperature: {temperature} (fixed)")
    
    st.sidebar.markdown("---")
    
    # Patient Persona Selection
    st.sidebar.markdown("### 👤 Patient Persona")
    
    if DEPENDENCIES_AVAILABLE:
        try:
            personas = load_patient_personas()
            persona_options = [f"Patient {i+1}: {p['ground_truth']}" for i, p in enumerate(personas)]
            
            selected_persona_idx = st.sidebar.selectbox(
                "Select Patient Scenario:",
                range(len(personas)),
                format_func=lambda x: persona_options[x]
            )
            
            selected_persona = personas[selected_persona_idx]
            
            st.sidebar.markdown("**Selected Persona:**")
            st.sidebar.text_area(
                "Description:",
                selected_persona['persona'],
                height=100,
                disabled=True,
                key=f"persona_{selected_persona_idx}"  # Add unique key
            )
            
            st.sidebar.markdown(f"**Ground Truth:** {selected_persona['ground_truth']}")
            
            # Store selected persona in session state for consistency
            st.session_state.selected_persona = selected_persona
            
        except Exception as e:
            st.sidebar.error(f"Error loading personas: {e}")
            selected_persona = None
    else:
        st.sidebar.error("Dependencies not available")
        selected_persona = None
    
    return api_key, "Google Gemini", model, temperature, selected_persona

def perform_safety_check(patient_response: str, interviewer_question: str, llm):
    """
    Perform real-time safety monitoring on patient response.
    
    Args:
        patient_response: What the patient said
        interviewer_question: The question that was asked
        llm: The LLM instance to use
        
    Returns:
        Safety assessment string
    """
    try:
        safety_prompt = f"""You are a SAFETY MONITOR for a medical interview. Your job is to immediately flag any safety concerns.

CRITICAL SAFETY INDICATORS TO FLAG:
🚨 IMMEDIATE DANGER:
- Suicidal thoughts, plans, or ideation
- Self-harm thoughts or behaviors  
- Thoughts of harming others
- Active psychosis or severe symptoms

⚠️ HIGH RISK:
- Substance abuse mentions
- Severe depression/hopelessness
- Loss of control
- Isolation/lack of support

RESPONSE FORMAT:
If NO safety concerns: "SAFE"
If safety concerns detected: "RISK: [LEVEL] - [SPECIFIC CONCERN] - [IMMEDIATE ACTION NEEDED]"

LEVELS: LOW, MODERATE, HIGH, CRITICAL

Examples:
- "RISK: CRITICAL - Active suicidal ideation with plan - IMMEDIATE INTERVENTION REQUIRED"
- "RISK: HIGH - Thoughts of death without plan - Monitor closely, consider safety plan"
- "RISK: MODERATE - Severe hopelessness - Follow up assessment needed"
- "SAFE"

Be VERY sensitive to safety language even if subtle.

Patient just said: "{patient_response}"

In context of question: "{interviewer_question}"

SAFETY ASSESSMENT:"""

        response = llm.invoke(safety_prompt)
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        logger.error(f"Safety check error: {e}")
        return "SAFE"  # Default to safe if error

def create_llm(provider: str, model: str, temperature: float):
    """Create Gemini LLM instance."""
    try:
        if not GEMINI_AVAILABLE:
            st.error("❌ **Gemini not installed!**\n\nRun this command:\n```bash\npip install langchain-google-genai\n```\nThen restart the app.")
            return None
        
        # Check if API key is set
        api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ Gemini API key not found. Please enter it in the sidebar.")
            return None
        
        # Create Gemini LLM with explicit API key
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=1000,
            timeout=30,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
        
    except Exception as e:
        st.error(f"❌ Error creating Gemini LLM: {e}")
        st.info("💡 **Tip**: Make sure you have a valid Google AI API key and the package is installed")
        logger.error(f"LLM creation error: {e}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'interview_state' not in st.session_state:
        st.session_state.interview_state = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if 'interview_complete' not in st.session_state:
        st.session_state.interview_complete = False
    
    if 'judge_evaluation' not in st.session_state:
        st.session_state.judge_evaluation = None
        
    if 'auto_progress' not in st.session_state:
        st.session_state.auto_progress = True
        
    if 'last_step_time' not in st.session_state:
        st.session_state.last_step_time = None

def display_conversation():
    """Display the conversation history."""
    st.markdown("## 💬 Interview Conversation")
    
    if not st.session_state.conversation_history:
        st.info("👋 Click 'Start Interview' to begin the conversation!")
        return
    
    # Create conversation container  
    with st.container():
        for i, entry in enumerate(st.session_state.conversation_history):
            timestamp = entry.get('timestamp', datetime.now().strftime("%H:%M:%S"))
            speaker = entry['speaker']
            message = entry['message']
            
            if speaker == "INTERVIEWER":
                st.markdown(f"""
                <div class="interviewer-msg">
                    <strong>🩺 Interviewer [{timestamp}]:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            
            elif speaker == "PATIENT":
                st.markdown(f"""
                <div class="patient-msg">
                    <strong>👤 Patient [{timestamp}]:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            
            elif speaker == "SYSTEM" or speaker == "system":
                st.markdown(f"""
                <div class="judge-msg">
                    <strong>🤖 System [{timestamp}]:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            
            elif speaker == "SAFETY_ALERT":
                # Show safety alerts prominently
                details = entry.get('details', 'Monitor closely')
                st.error(f"""
                **🚨 SAFETY ALERT [{timestamp}]**
                
                {message}
                
                **Action Required:** {details}
                """)
                
                # Also add to message flow for visual continuity
                st.markdown(f"""
                <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <strong>🚨 Safety Monitor [{timestamp}]:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)

def display_status_panel():
    """Display current interview status."""
    if st.session_state.interview_state is None:
        return
    
    st.markdown("## 📊 Interview Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = st.session_state.interview_state.get('interview_status', 'unknown')
        if status == 'completed':
            st.success("✅ **Status**\n\nCompleted")
        elif status == 'in_progress':
            st.info("🔄 **Status**\n\nIn Progress")
        else:
            st.info(f"📋 **Status**\n\n{status.title()}")
    
    with col2:
        risk_level = st.session_state.interview_state.get('risk_level', 'low')
        if risk_level == 'critical':
            st.error("🚨 **Risk Level**\n\nCritical")
        elif risk_level == 'high':
            st.warning("⚠️ **Risk Level**\n\nHigh")
        else:
            st.success(f"✅ **Risk Level**\n\n{risk_level.title()}")
    
    with col3:
        questions_answered = len(st.session_state.interview_state.get('symptom_map', {}))
        st.info(f"📈 **Questions**\n\n{questions_answered} Answered")
    
    with col4:
        positive_symptoms = len([k for k, v in st.session_state.interview_state.get('symptom_map', {}).items() if v])
        st.info(f"✅ **Symptoms**\n\n{positive_symptoms} Positive")
    
    # Safety flags - Enhanced display
    safety_flags = st.session_state.interview_state.get('safety_flags', [])
    if safety_flags:
        st.markdown("### 🚨 Safety Alerts")
        
        # Count different risk levels
        critical_flags = [f for f in safety_flags if "CRITICAL" in f.upper()]
        high_flags = [f for f in safety_flags if "HIGH" in f.upper()]
        other_flags = [f for f in safety_flags if "CRITICAL" not in f.upper() and "HIGH" not in f.upper()]
        
        # Display by priority
        for flag in critical_flags:
            st.error(f"🚨 CRITICAL: {flag}")
        for flag in high_flags:
            st.warning(f"⚠️ HIGH RISK: {flag}")
        for flag in other_flags:
            st.info(f"📋 NOTE: {flag}")
            
        # Safety summary
        if critical_flags:
            st.error("⚠️ **IMMEDIATE INTERVENTION MAY BE REQUIRED**")
        elif high_flags:
            st.warning("⚠️ **CLOSE MONITORING RECOMMENDED**")

def display_judge_evaluation():
    """Display judge evaluation if available."""
    if st.session_state.judge_evaluation is None:
        return
    
    st.markdown("## ⚖️ Judge Evaluation")
    
    # Create expandable section for detailed evaluation
    with st.expander("📋 Detailed Performance Analysis", expanded=True):
        st.markdown(st.session_state.judge_evaluation)

def run_interview_step(interviewer_agent, patient_agent, persona, llm):
    """Run one step of the interview - simplified version."""
    try:
        # Configuration for the graph
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        # Step 1: Get question from interviewer agent
        result = interviewer_agent.invoke(st.session_state.interview_state, config)
        st.session_state.interview_state = result
        
        # Check if interview is complete
        from agents import is_interview_complete, generate_diagnosis
        if is_interview_complete(result):
            st.session_state.interview_complete = True
            
            # Generate simple patient summary
            patient_summary = generate_diagnosis(result)
            st.session_state.interview_state["final_diagnosis"] = patient_summary
            
            # Add completion message
            st.session_state.conversation_history.append({
                'speaker': 'SYSTEM',
                'message': f"Interview completed. Patient summary: {patient_summary}",
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            return True  # Interview complete
        
        # Check if there's a new question in the transcript
        if result.get("transcript") and len(result["transcript"]) > 0:
            latest_entry = result["transcript"][-1]
            
            # Only process if it's a new interviewer question we haven't seen
            if "INTERVIEWER:" in latest_entry or "interviewer:" in latest_entry:
                # Extract question text
                if "INTERVIEWER:" in latest_entry:
                    question = latest_entry.split("INTERVIEWER:", 1)[1].strip()
                else:
                    question = latest_entry.split("interviewer:", 1)[1].strip()
                
                # Check if this question is already in our conversation history
                last_interviewer_msg = None
                for msg in reversed(st.session_state.conversation_history):
                    if msg['speaker'] == 'INTERVIEWER':
                        last_interviewer_msg = msg['message']
                        break
                
                # Only add if it's a new question
                if last_interviewer_msg != question:
                    # Add interviewer question to conversation
                    st.session_state.conversation_history.append({
                        'speaker': 'INTERVIEWER',
                        'message': question,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Get patient response - ensure persona is being used correctly
                    patient_response = patient_agent.invoke({
                        "persona": persona['persona'],
                        "question": question
                    })
                    
                    # Log for debugging
                    logger.info(f"Patient persona: {persona['persona'][:100]}...")
                    logger.info(f"Question asked: {question}")
                    logger.info(f"Patient response: {patient_response[:100]}...")
                    
                    # Add patient response to conversation
                    st.session_state.conversation_history.append({
                        'speaker': 'PATIENT',
                        'message': patient_response,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # SAFETY MONITORING - Check for dangerous content immediately
                    try:
                        with st.spinner("🔍 Safety monitoring..."):
                            # Use the same LLM that's already working for the interview
                            if llm is not None:
                                safety_assessment = perform_safety_check(patient_response, question, llm)
                            else:
                                safety_assessment = "SAFE"  # Skip if no LLM available
                        
                        # Process safety assessment
                        if safety_assessment and "RISK:" in safety_assessment:
                            # Extract risk level and details
                            risk_parts = safety_assessment.split(" - ")
                            risk_level = risk_parts[0].replace("RISK: ", "").strip()
                            risk_concern = risk_parts[1] if len(risk_parts) > 1 else "Safety concern detected"
                            action_needed = risk_parts[2] if len(risk_parts) > 2 else "Monitor closely"
                            
                            # Add safety alert to conversation
                            st.session_state.conversation_history.append({
                                'speaker': 'SAFETY_ALERT',
                                'message': f"🚨 SAFETY ALERT: {risk_level} - {risk_concern}",
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'details': action_needed
                            })
                            
                            # Update interview state with safety flag
                            from utils import add_safety_flag, update_risk_level, RiskLevel
                            st.session_state.interview_state = add_safety_flag(
                                st.session_state.interview_state, 
                                f"{risk_level}: {risk_concern}"
                            )
                            
                            # Update risk level based on assessment
                            if "CRITICAL" in risk_level.upper():
                                st.session_state.interview_state = update_risk_level(st.session_state.interview_state, RiskLevel.CRITICAL)
                            elif "HIGH" in risk_level.upper():
                                st.session_state.interview_state = update_risk_level(st.session_state.interview_state, RiskLevel.HIGH)
                                
                            # Show immediate alert
                            st.success(f"🚨 Safety Alert Triggered: {risk_level}")
                                
                    except Exception as safety_error:
                        logger.error(f"Safety monitoring error: {safety_error}")
                        st.error(f"⚠️ Safety monitoring error: {safety_error}")
                    
                    # Add patient response to interview state
                    from utils import add_to_transcript
                    st.session_state.interview_state = add_to_transcript(
                        st.session_state.interview_state, 
                        "user", 
                        patient_response
                    )
                    
                    # Parse response and route to next question
                    from agents import parse_response_node, route_logic_node
                    if llm is not None:
                        parsed_response = parse_response_node(st.session_state.interview_state, llm)
                    else:
                        parsed_response = parse_response_node(st.session_state.interview_state)  # Use fallback method
                    st.session_state.interview_state = route_logic_node(st.session_state.interview_state, parsed_response)
        
        return False  # Continue interview
        
    except Exception as e:
        st.error(f"❌ Error in interview step: {e}")
        return True  # Stop on error

def main():
    """Main Streamlit application."""
    # Title
    st.title("🏥 Three-Agent Medical Interview System")
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        st.error(f"""
        ❌ **Missing Dependencies**
        
        Please install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
        
        Error: {IMPORT_ERROR}
        """)
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    api_key, provider, model, temperature, selected_persona = setup_sidebar()
    
    # Show architecture
    show_architecture()
    
    # Check if we can proceed
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to continue.")
        return
    
    if not selected_persona:
        st.error("❌ Unable to load patient personas. Check the console for errors.")
        return
    
    # Main interface
    st.markdown("---")
    
    # Auto-progression is now automatic - no user control needed
    speed = 2  # Hardcoded 2 second delay
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Interview", disabled=st.session_state.interview_complete):
            try:
                # Test LLM connection first
                with st.spinner("🔍 Testing Gemini connection..."):
                    llm = create_llm(provider, model, temperature)
                    if llm is None:
                        return
                    
                    # Quick test to verify API key works
                    try:
                        test_response = llm.invoke("Test")
                        st.success("✅ Gemini connection successful!")
                    except Exception as test_error:
                        st.error(f"❌ Gemini API test failed: {test_error}")
                        if "authentication" in str(test_error).lower() or "api" in str(test_error).lower() or "key" in str(test_error).lower():
                            st.error("🔑 Please check your API key")
                        return
                
                # Initialize agents
                interviewer_agent = build_interviewer_graph()
                patient_agent = create_patient_agent(llm)
                judge_agent = create_judge_agent(llm)
                
                # Initialize interview state
                st.session_state.interview_state = create_initial_state(
                    session_id=st.session_state.session_id,
                    patient_persona=selected_persona['persona']
                )
                
                st.session_state.conversation_history = []
                st.session_state.interview_complete = False
                st.session_state.last_step_time = None  # Set to None so first step triggers immediately
                st.session_state.auto_progress = True  # Enable automatic progression
                
                st.success("✅ Interview started! The interview will progress automatically.")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error starting interview: {e}")
                logger.error(f"Start interview error: {e}")
    
    # Automatic progression logic - no manual button needed
    with col2:
        # Show progression status
        if st.session_state.interview_state and not st.session_state.interview_complete:
            if st.session_state.auto_progress and st.session_state.last_step_time:
                time_since_last = time.time() - st.session_state.last_step_time
                if time_since_last < speed:
                    remaining = speed - time_since_last
                    st.info(f"⏱️ Next step in {remaining:.1f} seconds...")
                else:
                    st.info("🔄 Processing next step...")
            else:
                st.info("🔄 Interview in progress...")
        elif st.session_state.interview_complete:
            st.success("✅ Interview Complete")
        
        # Auto-progression logic
        should_auto_progress = False
        if (st.session_state.auto_progress and 
            st.session_state.interview_state is not None and 
            not st.session_state.interview_complete):
            
            if st.session_state.last_step_time is None:
                should_auto_progress = True  # First step
                st.info("🚀 Starting first interview step...")
            else:
                time_since_last = time.time() - st.session_state.last_step_time
                if time_since_last >= speed:
                    should_auto_progress = True
                    st.info("⏰ Time elapsed, proceeding to next step...")
        
        if should_auto_progress:
            try:
                llm = create_llm(provider, model, temperature)
                if llm is None:
                    return
                
                interviewer_agent = build_interviewer_graph()
                patient_agent = create_patient_agent(llm)
                
                # Use the persona from session state if available, otherwise use selected_persona
                current_persona = st.session_state.get('selected_persona', selected_persona)
                st.info("🔄 Running interview step...")
                complete = run_interview_step(interviewer_agent, patient_agent, current_persona, llm)
                st.session_state.last_step_time = time.time()
                st.info(f"✅ Interview step completed. Complete: {complete}")
                
                # If interview is complete, run judge evaluation
                if complete and st.session_state.judge_evaluation is None:
                    judge_agent = create_judge_agent(llm)
                    transcript_text = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in st.session_state.conversation_history])
                    
                    # Use the persona from session state if available
                    current_persona = st.session_state.get('selected_persona', selected_persona)
                    
                    evaluation = judge_agent.invoke({
                        "ground_truth_persona": current_persona['ground_truth'],
                        "interview_transcript": transcript_text,
                        "final_diagnosis": st.session_state.interview_state.get("final_diagnosis", "")
                    })
                    
                    st.session_state.judge_evaluation = evaluation
                    st.session_state.auto_progress = False  # Stop auto-progression when complete
                
                # Auto-rerun for next step if auto-progression is on (but limit frequency)
                if st.session_state.auto_progress and not complete and not st.session_state.interview_complete:
                    time.sleep(0.5)  # Slower to prevent infinite loops
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error in interview step: {e}")
                st.session_state.auto_progress = False
    
    with col3:
        # Reset and Save buttons in one column
        col3_sub1, col3_sub2 = st.columns(2)
        
        with col3_sub1:
            if st.button("🔄 Reset Interview"):
                st.session_state.conversation_history = []
                st.session_state.interview_state = None
                st.session_state.interview_complete = False
                st.session_state.judge_evaluation = None
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.last_step_time = None
                st.session_state.auto_progress = False
                st.success("✅ Interview reset!")
                st.rerun()
        
        with col3_sub2:
            if st.button("💾 Save Session") and st.session_state.interview_state:
                try:
                    session_data = {
                        "session_id": st.session_state.session_id,
                        "persona": selected_persona,
                        "conversation": st.session_state.conversation_history,
                        "final_state": st.session_state.interview_state,
                        "judge_evaluation": st.session_state.judge_evaluation,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    filename = f"session_{st.session_state.session_id}.json"
                    with open(filename, 'w') as f:
                        json.dump(session_data, f, indent=2, default=str)
                    
                    st.success(f"✅ Session saved to {filename}")
                    
                except Exception as e:
                    st.error(f"❌ Error saving session: {e}")
    

    
    # Safety banner - show at top if there are safety concerns
    if (st.session_state.interview_state and 
        st.session_state.interview_state.get('safety_flags', [])):
        
        safety_flags = st.session_state.interview_state.get('safety_flags', [])
        critical_flags = [f for f in safety_flags if "CRITICAL" in f.upper()]
        high_flags = [f for f in safety_flags if "HIGH" in f.upper()]
        
        if critical_flags:
            st.error(f"""
            🚨 **CRITICAL SAFETY ALERT** 🚨
            
            {len(critical_flags)} critical safety concern(s) detected.
            
            **IMMEDIATE INTERVENTION MAY BE REQUIRED**
            """)
        elif high_flags:
            st.warning(f"""
            ⚠️ **HIGH RISK SAFETY ALERT** ⚠️
            
            {len(high_flags)} high-risk safety concern(s) detected.
            
            **CLOSE MONITORING RECOMMENDED**
            """)
    
    # Display main content
    display_status_panel()
    
    # Debug panel for development
    if st.session_state.interview_state:
        with st.expander("🔍 Debug Info", expanded=False):
            safety_flags = st.session_state.interview_state.get("safety_flags", [])
            debug_data = {
                "current_question_id": st.session_state.interview_state.get("current_question_id"),
                "interview_status": st.session_state.interview_state.get("interview_status"),
                "risk_level": st.session_state.interview_state.get("risk_level"),
                "symptom_map": st.session_state.interview_state.get("symptom_map", {}),
                "safety_flags_count": len(safety_flags),
                "conversation_count": len(st.session_state.conversation_history)
            }
            st.json(debug_data)
            
            # Show symptom summary
            symptom_map = st.session_state.interview_state.get("symptom_map", {})
            if symptom_map:
                st.markdown("**📋 Symptom Summary:**")
                positive_symptoms = [k for k, v in symptom_map.items() if v]
                negative_symptoms = [k for k, v in symptom_map.items() if not v]
                
                col_debug1, col_debug2 = st.columns(2)
                with col_debug1:
                    st.success(f"✅ **Positive:** {', '.join(positive_symptoms) if positive_symptoms else 'None'}")
                with col_debug2:
                    st.info(f"❌ **Negative:** {', '.join(negative_symptoms) if negative_symptoms else 'None'}")
                
                # Test safety monitoring
                st.markdown("**🧪 Test Safety Monitor:**")
                test_response = st.text_input("Test patient response:", key="safety_test")
                if st.button("🔍 Test Safety Check", key="test_safety") and test_response:
                    try:
                        llm = create_llm(provider, model, temperature)
                        if llm:
                            safety_result = perform_safety_check(
                                test_response, 
                                "Test question about safety", 
                                llm
                            )
                            if "RISK:" in safety_result:
                                st.error(f"🚨 **Safety Alert:** {safety_result}")
                            else:
                                st.success(f"✅ **Safe:** {safety_result}")
                    except Exception as e:
                        st.error(f"Test error: {e}")
    
    display_conversation()
    display_judge_evaluation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; margin-top: 20px;'>
            🏥 Three-Agent Medical Interview System | Built with Streamlit, LangChain & LangGraph
        </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh is handled in the main UI logic above

if __name__ == "__main__":
    main() 