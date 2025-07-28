#!/usr/bin/env python3
"""
Main Simulation Script for Three-Agent Medical Interview System

This script orchestrates the interaction between:
- Interviewer Agent (LangGraph-based question flow)
- Patient Agent (LangChain runnable for simulated responses) 
- Judge Agent (LangChain runnable for performance evaluation)
"""

import os
import json
import uuid
from typing import Dict, Any, List
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

# LangGraph imports  
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from utils import (
    AgentState, create_initial_state, add_to_transcript, 
    is_interview_complete, serialize_state_for_db, InterviewStatus
)
from agents import (
    get_question_node, parse_response_node, route_logic_node,
    create_patient_agent, create_judge_agent, generate_diagnosis,
    load_question_data
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_openai_llm(model: str = "gpt-4", temperature: float = 0.3) -> ChatOpenAI:
    """
    Setup OpenAI LLM with proper configuration.
    
    Args:
        model: OpenAI model name
        temperature: Temperature for response generation
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=1000,
        timeout=30
    )
    
    logger.info(f"Initialized OpenAI LLM: {model} (temp={temperature})")
    return llm

def load_patient_personas() -> List[Dict[str, str]]:
    """
    Define sample patient personas for testing.
    
    Returns:
        List of patient persona dictionaries
    """
    personas = [
        {
            'persona': 'A 45-year-old man who has been feeling profoundly sad and has lost interest in his hobbies for the past month. He has also been having trouble sleeping and feels worthless. He has had some thoughts about death but no specific plans.',
            'ground_truth': 'Meets criteria for Major Depressive Episode'
        },
        {
            'persona': 'A 28-year-old woman who has been feeling down for about a week after losing her job, but still enjoys spending time with friends and has normal sleep patterns. She feels this is situational and temporary.',
            'ground_truth': 'Does not meet criteria for Major Depressive Episode - situational stress'
        },
        {
            'persona': 'A 35-year-old teacher who has been experiencing severe depression for three weeks including sadness, loss of interest, insomnia, fatigue, guilt, concentration problems, and persistent thoughts of suicide with a specific plan.',
            'ground_truth': 'Meets criteria for Major Depressive Episode with high suicide risk - requires immediate intervention'
        },
        {
            'persona': 'A 22-year-old college student who feels generally happy but has been having some sleep issues due to exam stress. No significant mood changes or loss of interest in activities.',
            'ground_truth': 'Does not meet criteria for Major Depressive Episode - normal stress response'
        }
    ]
    
    return personas

def build_interviewer_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the interviewer agent.
    
    Returns:
        Compiled LangGraph interviewer agent
    """
    logger.info("Building interviewer graph...")
    
    # Initialize StateGraph with our AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes for the interviewer agent
    workflow.add_node("get_question", get_question_node)
    workflow.add_node("parse_response", lambda state: {
        **state, 
        "parsed_response": parse_response_node(state)
    })
    workflow.add_node("route_logic", lambda state: route_logic_node(state, state.get("parsed_response", "no")))
    
    # Set entry point
    workflow.set_entry_point("get_question")
    
    # Define the flow edges
    workflow.add_edge("get_question", "parse_response")
    workflow.add_edge("parse_response", "route_logic")
    
    # Add conditional edge for loop or completion
    workflow.add_conditional_edges(
        "route_logic",
        lambda state: "complete" if is_interview_complete(state) else "continue",
        {
            "continue": "get_question",
            "complete": END
        }
    )
    
    # Compile with memory checkpoint
    memory = MemorySaver()
    interviewer_agent = workflow.compile(checkpointer=memory)
    
    logger.info("Interviewer graph compiled successfully")
    return interviewer_agent

def print_conversation_header(persona: Dict[str, str], session_id: str):
    """Print formatted conversation header."""
    print("\n" + "="*80)
    print(f"🏥 MEDICAL INTERVIEW SIMULATION - Session: {session_id}")
    print("="*80)
    print(f"📋 Patient Persona: {persona['persona']}")
    print(f"🎯 Ground Truth: {persona['ground_truth']}")
    print("="*80)
    print()

def print_conversation_turn(speaker: str, message: str, turn_number: int):
    """Print formatted conversation turn."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if speaker.upper() == "INTERVIEWER":
        emoji = "🩺"
        color = "\033[94m"  # Blue
    elif speaker.upper() == "PATIENT":
        emoji = "👤"
        color = "\033[92m"  # Green
    else:
        emoji = "🤖"
        color = "\033[93m"  # Yellow
    
    reset = "\033[0m"
    
    print(f"{color}[{timestamp}] Turn {turn_number} - {emoji} {speaker.upper()}:{reset}")
    print(f"  {message}")
    print()

def run_simulation(
    interviewer_agent: StateGraph,
    patient_agent: Runnable,
    judge_agent: Runnable,
    persona: Dict[str, str],
    max_turns: int = 20
) -> Dict[str, Any]:
    """
    Run a complete interview simulation.
    
    Args:
        interviewer_agent: Compiled LangGraph interviewer
        patient_agent: Patient simulation runnable
        judge_agent: Judge evaluation runnable  
        persona: Patient persona dictionary
        max_turns: Maximum conversation turns
        
    Returns:
        Simulation results dictionary
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Print conversation header
    print_conversation_header(persona, session_id)
    
    # Initialize state
    initial_state = create_initial_state(
        session_id=session_id,
        patient_persona=persona['persona']
    )
    
    # Configuration for the graph
    config = {"configurable": {"thread_id": session_id}}
    
    turn_number = 0
    
    try:
        logger.info(f"Starting simulation for session {session_id}")
        
        # Main simulation loop
        while turn_number < max_turns:
            turn_number += 1
            
            # Step 1: Get question from interviewer agent
            logger.debug(f"Turn {turn_number}: Getting question from interviewer")
            result = interviewer_agent.invoke(initial_state, config)
            
            # Extract the latest question from transcript
            if result["transcript"]:
                latest_entry = result["transcript"][-1]
                if "INTERVIEWER:" in latest_entry:
                    question = latest_entry.split("INTERVIEWER:", 1)[1].strip()
                    print_conversation_turn("INTERVIEWER", question, turn_number)
                    
                    # Check if interview is complete
                    if is_interview_complete(result):
                        logger.info("Interview completed")
                        break
                        
                    # Step 2: Get patient response
                    logger.debug(f"Turn {turn_number}: Getting patient response")
                    patient_response = patient_agent.invoke({
                        "persona": persona['persona'],
                        "question": question
                    })
                    
                    print_conversation_turn("PATIENT", patient_response, turn_number)
                    
                    # Step 3: Add patient response to state and continue
                    result = add_to_transcript(result, "user", patient_response)
                    initial_state = result
                    
            else:
                logger.warning(f"No transcript found in turn {turn_number}")
                break
        
        # Generate final diagnosis
        logger.info("Generating final diagnosis...")
        final_diagnosis = generate_diagnosis(result)
        result["final_diagnosis"] = final_diagnosis
        
        print("\n" + "="*60)
        print("🏁 INTERVIEW COMPLETED")
        print("="*60)
        print(f"📊 Final Status: {result['interview_status']}")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        print(f"🩺 Diagnosis: {final_diagnosis}")
        
        if result['safety_flags']:
            print(f"🚨 Safety Flags: {', '.join(result['safety_flags'])}")
        
        print(f"📈 Questions Answered: {len(result['symptom_map'])}")
        print(f"✅ Positive Symptoms: {len([k for k, v in result['symptom_map'].items() if v])}")
        print()
        
        # Step 4: Judge evaluation
        logger.info("Running judge evaluation...")
        print("="*60)
        print("⚖️  JUDGE EVALUATION")
        print("="*60)
        
        # Prepare transcript for judge
        transcript_text = "\n".join(result["transcript"])
        
        judge_evaluation = judge_agent.invoke({
            "ground_truth_persona": persona['ground_truth'],
            "interview_transcript": transcript_text,
            "final_diagnosis": final_diagnosis
        })
        
        print(judge_evaluation)
        print("\n" + "="*80)
        
        # Return comprehensive results
        return {
            "session_id": session_id,
            "persona": persona,
            "final_state": result,
            "final_diagnosis": final_diagnosis,
            "judge_evaluation": judge_evaluation,
            "total_turns": turn_number,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        print(f"\n❌ SIMULATION ERROR: {e}")
        return {
            "session_id": session_id,
            "persona": persona,
            "error": str(e),
            "success": False
        }

def main():
    """Main function to run the three-agent simulation."""
    try:
        print("🚀 Starting Three-Agent Medical Interview Simulation")
        print("="*60)
        
        # 1. Setup OpenAI LLM
        print("⚙️  Setting up OpenAI LLM...")
        llm = setup_openai_llm()
        
        # 2. Load question data
        print("📋 Loading interview questions...")
        question_data = load_question_data()
        logger.info(f"Loaded {len(question_data['questions'])} questions")
        
        # 3. Load patient personas
        print("👥 Loading patient personas...")
        personas = load_patient_personas()
        logger.info(f"Loaded {len(personas)} patient personas")
        
        # 4. Create agent instances
        print("🤖 Creating agent instances...")
        patient_agent = create_patient_agent(llm)
        judge_agent = create_judge_agent(llm)
        
        # 5. Build interviewer graph
        print("🔧 Building interviewer graph...")
        interviewer_agent = build_interviewer_graph()
        
        print("\n✅ All components initialized successfully!")
        
        # 6. Run simulations
        results = []
        
        for i, persona in enumerate(personas, 1):
            print(f"\n🎬 Running Simulation {i}/{len(personas)}")
            
            result = run_simulation(
                interviewer_agent=interviewer_agent,
                patient_agent=patient_agent,
                judge_agent=judge_agent,
                persona=persona,
                max_turns=25
            )
            
            results.append(result)
            
            # Save intermediate results
            if result["success"]:
                # Save session data (in real app, this would go to database)
                session_file = f"session_{result['session_id']}.json"
                with open(session_file, 'w') as f:
                    json.dump(serialize_state_for_db(result["final_state"]), f, indent=2)
                logger.info(f"Session data saved to {session_file}")
        
        # 7. Summary report
        print("\n" + "="*80)
        print("📊 SIMULATION SUMMARY REPORT")
        print("="*80)
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"✅ Successful simulations: {len(successful)}/{len(results)}")
        if failed:
            print(f"❌ Failed simulations: {len(failed)}")
            for failure in failed:
                print(f"   - Session {failure['session_id']}: {failure['error']}")
        
        if successful:
            avg_turns = sum(r["total_turns"] for r in successful) / len(successful)
            print(f"📈 Average conversation turns: {avg_turns:.1f}")
            
            risk_levels = [r["final_state"]["risk_level"] for r in successful]
            print(f"⚠️  Risk level distribution: {dict(zip(*zip(*[(level, risk_levels.count(level)) for level in set(risk_levels)])))} ")
        
        print("\n🎉 Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"\n💥 Critical error: {e}")
        raise

if __name__ == "__main__":
    main()
