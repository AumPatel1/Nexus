from typing import TypedDict, Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewStatus(Enum):
    """Enum for tracking interview progress"""
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ERROR = "error"

class RiskLevel(Enum):
    """Enum for risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentState(TypedDict):
    """
    LangGraph AgentState for managing interview conversation state.
    This state is shared across all agents in the system.
    """
    patient_persona: str  # Brief description of the patient's condition
    transcript: List[str]  # Running log of the conversation
    current_question_id: str  # ID of the question to ask next (e.g., 'A1')
    symptom_map: Dict[str, bool]  # Track answers to screener questions {'A1': True, 'A2': False}
    final_diagnosis: str  # Final conclusion/diagnosis
    
    # Additional fields for enhanced functionality
    session_id: str  # Unique identifier for this interview session
    interview_status: str  # Current status of the interview (InterviewStatus enum)
    risk_level: str  # Current assessed risk level (RiskLevel enum)
    timestamp: str  # ISO timestamp of last update
    metadata: Dict[str, Any]  # Additional metadata (demographics, etc.)
    safety_flags: List[str]  # List of any safety concerns detected
    question_history: List[str]  # Track which questions have been asked
    confidence_scores: Dict[str, float]  # Confidence in various assessments

def create_initial_state(session_id: str, patient_persona: str = "") -> AgentState:
    """
    Create an initial AgentState for a new interview session.
    
    Args:
        session_id: Unique identifier for the session
        patient_persona: Optional initial patient description
        
    Returns:
        AgentState: Initial state dictionary
    """
    return AgentState(
        patient_persona=patient_persona,
        transcript=[],
        current_question_id="A1",  # Start with first question
        symptom_map={},
        final_diagnosis="",
        session_id=session_id,
        interview_status=InterviewStatus.STARTING.value,
        risk_level=RiskLevel.LOW.value,
        timestamp=datetime.now().isoformat(),
        metadata={},
        safety_flags=[],
        question_history=[],
        confidence_scores={}
    )

def update_state_timestamp(state: AgentState) -> AgentState:
    """Update the timestamp in the state to current time."""
    state["timestamp"] = datetime.now().isoformat()
    return state

def add_to_transcript(state: AgentState, speaker: str, message: str) -> AgentState:
    """
    Add a new entry to the conversation transcript.
    
    Args:
        state: Current agent state
        speaker: Who is speaking ('user', 'interviewer', 'system')
        message: The message content
        
    Returns:
        Updated state
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    transcript_entry = f"[{timestamp}] {speaker.upper()}: {message}"
    state["transcript"].append(transcript_entry)
    state = update_state_timestamp(state)
    return state

def update_symptom_map(state: AgentState, question_id: str, answer: bool) -> AgentState:
    """
    Update the symptom map with a new answer.
    
    Args:
        state: Current agent state
        question_id: The question identifier (e.g., 'A1', 'B2')
        answer: Boolean answer to the question
        
    Returns:
        Updated state
    """
    state["symptom_map"][question_id] = answer
    if question_id not in state["question_history"]:
        state["question_history"].append(question_id)
    state = update_state_timestamp(state)
    logger.info(f"Updated symptom map: {question_id} = {answer}")
    return state

def add_safety_flag(state: AgentState, flag: str) -> AgentState:
    """
    Add a safety flag to the state.
    
    Args:
        state: Current agent state
        flag: Description of the safety concern
        
    Returns:
        Updated state
    """
    if flag not in state["safety_flags"]:
        state["safety_flags"].append(flag)
        logger.warning(f"Safety flag added: {flag}")
    state = update_state_timestamp(state)
    return state

def update_risk_level(state: AgentState, risk_level: RiskLevel) -> AgentState:
    """
    Update the current risk assessment level.
    
    Args:
        state: Current agent state
        risk_level: New risk level assessment
        
    Returns:
        Updated state
    """
    old_level = state["risk_level"]
    state["risk_level"] = risk_level.value
    state = update_state_timestamp(state)
    
    if old_level != risk_level.value:
        logger.info(f"Risk level changed from {old_level} to {risk_level.value}")
        
    return state

def is_interview_complete(state: AgentState) -> bool:
    """
    Check if the interview should be considered complete.
    
    Args:
        state: Current agent state
        
    Returns:
        True if interview is complete, False otherwise
    """
    # Interview is complete if status is completed or escalated
    return state["interview_status"] in [
        InterviewStatus.COMPLETED.value,
        InterviewStatus.ESCALATED.value
    ]

def get_answered_questions_count(state: AgentState) -> int:
    """Get the number of questions that have been answered."""
    return len(state["symptom_map"])

def get_positive_symptoms(state: AgentState) -> Dict[str, bool]:
    """Get all symptoms that were answered as True."""
    return {k: v for k, v in state["symptom_map"].items() if v}

def get_negative_symptoms(state: AgentState) -> Dict[str, bool]:
    """Get all symptoms that were answered as False."""
    return {k: v for k, v in state["symptom_map"].items() if not v}

def serialize_state_for_db(state: AgentState) -> Dict[str, Any]:
    """
    Serialize the AgentState for database storage.
    
    Args:
        state: AgentState to serialize
        
    Returns:
        Dictionary suitable for database storage
    """
    return {
        "session_id": state["session_id"],
        "patient_persona": state["patient_persona"],
        "transcript": state["transcript"],
        "current_question_id": state["current_question_id"],
        "symptom_map": state["symptom_map"],
        "final_diagnosis": state["final_diagnosis"],
        "interview_status": state["interview_status"],
        "risk_level": state["risk_level"],
        "timestamp": state["timestamp"],
        "metadata": state["metadata"],
        "safety_flags": state["safety_flags"],
        "question_history": state["question_history"],
        "confidence_scores": state["confidence_scores"],
        "total_questions_answered": get_answered_questions_count(state),
        "positive_symptoms": get_positive_symptoms(state),
        "negative_symptoms": get_negative_symptoms(state)
    }

def validate_state(state: AgentState) -> bool:
    """
    Validate that the AgentState has all required fields and proper types.
    
    Args:
        state: AgentState to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "patient_persona", "transcript", "current_question_id", 
        "symptom_map", "final_diagnosis", "session_id", 
        "interview_status", "risk_level", "timestamp"
    ]
    
    try:
        for field in required_fields:
            if field not in state:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Type checks
        if not isinstance(state["transcript"], list):
            logger.error("transcript must be a list")
            return False
            
        if not isinstance(state["symptom_map"], dict):
            logger.error("symptom_map must be a dict")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"State validation error: {e}")
        return False
