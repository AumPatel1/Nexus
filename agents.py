import json
import logging
from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from utils import (
    AgentState, add_to_transcript, update_symptom_map, 
    add_safety_flag, update_risk_level, RiskLevel, InterviewStatus
)

# Configure logging
logger = logging.getLogger(__name__)

# Load the question flow logic
def load_question_data() -> Dict[str, Any]:
    """Load the question flow data from mini_module_a.json"""
    try:
        with open('mini_module_a.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("mini_module_a.json file not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing mini_module_a.json: {e}")
        raise

# =============================================================================
# INTERVIEWER AGENT NODES (LangGraph Functions)
# =============================================================================

def get_question_node(state: AgentState) -> AgentState:
    """
    LangGraph node that reads the question data and appends the current question to transcript.
    
    Args:
        state: Current AgentState
        
    Returns:
        Updated AgentState with question added to transcript
    """
    try:
        question_data = load_question_data()
        current_id = state["current_question_id"]
        
        # Get the question from the questions section
        question_info = question_data["questions"].get(current_id)
        if not question_info:
            logger.error(f"Question ID {current_id} not found in question data")
            message = "I'm sorry, there seems to be an issue with the interview flow."
            state = add_safety_flag(state, f"Unknown question ID: {current_id}")
        else:
            message = question_info["question_text"]
            
            # Check for endpoint states
            if current_id.startswith("END_MODULE"):
                state["interview_status"] = InterviewStatus.COMPLETED.value
                if current_id == "END_MODULE_SUCCESS":
                    message = "Thank you for completing the screening. Based on your responses, I will now provide an assessment."
                else:  # END_MODULE_FAIL
                    message = "Based on your responses, you don't appear to meet the criteria for major depression at this time."
            
            # Add safety monitoring for suicide-related questions
            if "suicide" in message.lower() or "death" in message.lower() or "harm" in message.lower():
                state = update_risk_level(state, RiskLevel.HIGH)
        
        # Add question to transcript
        state = add_to_transcript(state, "interviewer", message)
        logger.info(f"Asked question {current_id}: {message[:50]}...")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in get_question_node: {e}")
        error_message = "I'm sorry, there was a technical issue. Let me try to continue."
        state = add_to_transcript(state, "interviewer", error_message)
        state = add_safety_flag(state, f"get_question_node error: {str(e)}")
        return state

def parse_response_node(state: AgentState) -> str:
    """
    LangGraph node that uses OpenAI API with function calling to classify user response as yes/no.
    
    Args:
        state: Current AgentState
        
    Returns:
        Parsed response as 'yes' or 'no' string
    """
    try:
        if not state["transcript"]:
            logger.warning("No transcript available for parsing")
            return "no"
        
        # Get the latest user response
        latest_response = None
        for entry in reversed(state["transcript"]):
            if entry.upper().startswith("[") and "USER:" in entry.upper():
                latest_response = entry.split("USER:", 1)[1].strip()
                break
        
        if not latest_response:
            logger.warning("No user response found in transcript")
            return "no"
        
        # Initialize OpenAI client
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Define the function for classification
        classification_function = {
            "name": "classify_response",
            "description": "Classify a patient's response to a medical screening question as yes or no",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Whether the patient's response indicates 'yes' or 'no' to the screening question"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence level in the classification (0-1)"
                    }
                },
                "required": ["classification"]
            }
        }
        
        # Create the prompt for classification
        system_prompt = """You are a medical interview assistant that classifies patient responses to screening questions.

Your task is to determine if the patient's response indicates "yes" or "no" to the screening question.

Guidelines:
- "Yes" responses: Affirmative answers, descriptions of symptoms, "sometimes", "a little", etc.
- "No" responses: Clear denials, "never", "not at all", "not really", etc.
- For ambiguous responses, err on the side of "yes" for safety in medical screening
- Consider context and implied meaning, not just literal words

Classify the response using the classify_response function."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Patient response: '{latest_response}'")
        ]
        
        # Make the API call with function calling
        response = llm.bind_functions([classification_function]).invoke(messages)
        
        # Extract the classification from function call
        if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
            function_call = response.additional_kwargs['function_call']
            if function_call['name'] == 'classify_response':
                result = json.loads(function_call['arguments'])
                classification = result['classification']
                confidence = result.get('confidence', 0.5)
                
                logger.info(f"Classified response '{latest_response[:30]}...' as '{classification}' (confidence: {confidence})")
                return classification
        
        # Fallback parsing if function calling fails
        response_lower = latest_response.lower()
        if any(word in response_lower for word in ['yes', 'yeah', 'sure', 'definitely', 'absolutely', 'sometimes', 'often', 'frequently']):
            return "yes"
        elif any(word in response_lower for word in ['no', 'not', 'never', 'rarely', 'hardly']):
            return "no"
        else:
            # Default to "yes" for safety in medical screening
            logger.warning(f"Ambiguous response, defaulting to 'yes' for safety: {latest_response}")
            return "yes"
            
    except Exception as e:
        logger.error(f"Error in parse_response_node: {e}")
        # Default to "no" if there's an error to avoid false positives
        return "no"

def route_logic_node(state: AgentState, parsed_response: str) -> AgentState:
    """
    LangGraph node that handles routing logic based on the parsed response.
    
    Args:
        state: Current AgentState  
        parsed_response: The classified 'yes' or 'no' response
        
    Returns:
        Updated AgentState with routing applied
    """
    try:
        question_data = load_question_data()
        current_id = state["current_question_id"]
        
        # Update symptom map if this was a screener question
        if current_id in question_data["questions"]:
            question_info = question_data["questions"][current_id]
            if question_info.get("is_screener", False):
                answer = parsed_response == "yes"
                state = update_symptom_map(state, current_id, answer)
        
        # Handle special CHECK_A1_A2 logic
        if current_id == "CHECK_A1_A2":
            # Check if either A1 or A2 is True in symptom_map
            a1_positive = state["symptom_map"].get("A1", False)
            a2_positive = state["symptom_map"].get("A2", False)
            
            if a1_positive or a2_positive:
                next_question = "A3"  # continue_if_one_yes
                logger.info("CHECK_A1_A2: At least one positive, continuing interview")
            else:
                next_question = "END_MODULE_FAIL"  # end_if_both_no
                logger.info("CHECK_A1_A2: Both negative, ending interview")
                
            state["current_question_id"] = next_question
            
        # Handle CHECK_DIAGNOSIS logic
        elif current_id == "CHECK_DIAGNOSIS":
            # Count positive symptoms (A1-A9)
            screener_symptoms = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
            positive_symptoms = sum(1 for q in screener_symptoms if state["symptom_map"].get(q, False))
            
            # Check core criteria: at least one of A1 or A2 must be positive
            has_core_symptoms = state["symptom_map"].get("A1", False) or state["symptom_map"].get("A2", False)
            
            # Check functional impairment (A10) and exclusion criteria (A11)
            has_impairment = state["symptom_map"].get("A10", False)
            has_exclusions = state["symptom_map"].get("A11", False)
            
            # Diagnosis criteria: 5+ symptoms, core symptoms present, impairment yes, exclusions no
            if (positive_symptoms >= 5 and has_core_symptoms and 
                has_impairment and not has_exclusions):
                next_question = "END_MODULE_SUCCESS"
                logger.info(f"Diagnosis met: {positive_symptoms} symptoms, core symptoms: {has_core_symptoms}")
            else:
                next_question = "END_MODULE_FAIL"
                logger.info(f"Diagnosis not met: {positive_symptoms} symptoms, core: {has_core_symptoms}, impairment: {has_impairment}, exclusions: {has_exclusions}")
                
            state["current_question_id"] = next_question
            
        # Handle regular question routing
        elif current_id in question_data["questions"]:
            question_info = question_data["questions"][current_id]
            skip_logic = question_info.get("skip_logic", {})
            
            if parsed_response == "yes":
                next_question = skip_logic.get("yes_destination", "END_MODULE_SUCCESS")
            else:
                next_question = skip_logic.get("no_destination", "END_MODULE_FAIL")
                
            state["current_question_id"] = next_question
            logger.info(f"Routed from {current_id} to {next_question} based on '{parsed_response}' response")
            
            # Special handling for high-risk responses
            if current_id == "A9" and parsed_response == "yes":
                state = update_risk_level(state, RiskLevel.CRITICAL)
                state = add_safety_flag(state, "Suicide/self-harm thoughts reported")
        
        # Handle endpoint states
        elif current_id.startswith("END_MODULE"):
            state["interview_status"] = InterviewStatus.COMPLETED.value
        
        return state
        
    except Exception as e:
        logger.error(f"Error in route_logic_node: {e}")
        state = add_safety_flag(state, f"route_logic_node error: {str(e)}")
        # Default to ending the interview on error
        state["current_question_id"] = "COMPLETE"
        state["interview_status"] = InterviewStatus.ERROR.value
        return state

# =============================================================================
# PATIENT AGENT (LangChain Runnable)
# =============================================================================

def create_patient_agent(llm: ChatOpenAI) -> Runnable:
    """
    Create a LangChain runnable for simulating patient responses.
    
    Args:
        llm: OpenAI language model instance
        
    Returns:
        LangChain runnable that generates patient responses
    """
    
    patient_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are simulating a patient in a medical interview based on the given persona.

Guidelines for your responses:
- Stay consistent with your persona throughout the conversation
- Give realistic, human-like answers that a real patient might give
- Vary your response length and style naturally
- Include some hesitation, uncertainty, or elaboration when appropriate
- Don't be overly clinical - speak like a regular person
- If you have the symptom, describe it in your own words
- If you don't have the symptom, clearly indicate that

Your persona: {persona}

Remember: You are role-playing as this patient. Respond naturally and authentically."""),
        ("human", "Interviewer: {question}\n\nRespond as the patient:")
    ])
    
    return patient_prompt | llm | StrOutputParser()

# =============================================================================
# JUDGE AGENT (LangChain Runnable)  
# =============================================================================

def create_judge_agent(llm: ChatOpenAI) -> Runnable:
    """
    Create a LangChain runnable for evaluating interviewer performance.
    
    Args:
        llm: OpenAI language model instance
        
    Returns:
        LangChain runnable that evaluates interview quality
    """
    
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert clinical supervisor evaluating a medical interview conducted by an AI system.

Your task is to provide a comprehensive evaluation of the interviewer's performance.

Evaluation criteria:
1. **Accuracy**: Did the final diagnosis match the ground truth persona?
2. **Completeness**: Were all relevant screening questions asked?
3. **Safety**: Were safety protocols followed for high-risk responses?
4. **Flow**: Was the interview logic and question sequencing appropriate?
5. **Communication**: Were questions clear and professional?

Provide your evaluation in the following format:

## Overall Performance: [Excellent/Good/Fair/Poor]

## Detailed Analysis:
- **Accuracy**: [Score 1-10] - [Explanation]
- **Completeness**: [Score 1-10] - [Explanation] 
- **Safety**: [Score 1-10] - [Explanation]
- **Flow**: [Score 1-10] - [Explanation]
- **Communication**: [Score 1-10] - [Explanation]

## Key Strengths:
- [List 2-3 strengths]

## Areas for Improvement:
- [List 2-3 areas needing improvement]

## Diagnostic Accuracy:
- Expected: [Based on persona]
- Actual: [From final diagnosis]
- Match: [Yes/No with explanation]

## Overall Score: [Average of all scores]/10"""),
        ("human", """Please evaluate this medical interview:

**Ground Truth Persona:** {ground_truth_persona}

**Interview Transcript:** {interview_transcript}

**Final Diagnosis:** {final_diagnosis}

Provide your detailed evaluation:""")
    ])
    
    return judge_prompt | llm | StrOutputParser()

# =============================================================================
# UTILITY FUNCTIONS FOR AGENTS
# =============================================================================

def get_next_question_id(state: AgentState) -> str:
    """Get the next question ID from the current state."""
    return state.get("current_question_id", "A1")

def is_interview_complete(state: AgentState) -> bool:
    """Check if the interview has reached a completion state."""
    current_id = state["current_question_id"]
    
    # Check if current question is an endpoint
    if current_id.startswith("END_MODULE"):
        return True
        
    # Check if interview status indicates completion
    if state["interview_status"] in [InterviewStatus.COMPLETED.value, InterviewStatus.ESCALATED.value]:
        return True
        
    return False

def generate_diagnosis(state: AgentState) -> str:
    """
    Generate a diagnosis based on the collected symptoms.
    
    Args:
        state: Current AgentState with symptom data
        
    Returns:
        Diagnosis string
    """
    try:
        positive_symptoms = [k for k, v in state["symptom_map"].items() if v]
        total_symptoms = len(positive_symptoms)
        
        # Basic depression screening logic (simplified)
        has_core_symptoms = state["symptom_map"].get("A1", False) or state["symptom_map"].get("A2", False)
        
        if state["symptom_map"].get("A9", False):  # Suicidal ideation
            return "Major Depressive Episode with Suicidal Ideation - IMMEDIATE INTERVENTION REQUIRED"
        elif has_core_symptoms and total_symptoms >= 5:
            return "Likely Major Depressive Episode - Recommend clinical evaluation"
        elif has_core_symptoms and total_symptoms >= 3:
            return "Possible depressive symptoms - Recommend further assessment"
        elif total_symptoms > 0:
            return "Some depressive symptoms present - Monitor and consider follow-up"
        else:
            return "No significant depressive symptoms detected"
            
    except Exception as e:
        logger.error(f"Error generating diagnosis: {e}")
        return "Unable to generate diagnosis due to technical error" 