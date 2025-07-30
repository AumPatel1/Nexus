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
        
        # Handle special logic nodes that shouldn't show questions
        if current_id == "CHECK_A1_A2":
            # This is internal logic, not a question - handle it directly
            a1_positive = state["symptom_map"].get("A1", False)
            a2_positive = state["symptom_map"].get("A2", False)
            
            logger.info(f"CHECK_A1_A2 Logic: A1={a1_positive}, A2={a2_positive}")
            logger.info(f"Full symptom_map: {state['symptom_map']}")
            
            # TEMPORARY DEBUG: Force continuation to test full interview
            if a1_positive or a2_positive or len(state["symptom_map"]) > 0:
                state["current_question_id"] = "A3"
                logger.info("CHECK_A1_A2: Continuing to A3 (forced for debugging)")
                # Add a system message to transcript for debugging
                state = add_to_transcript(state, "system", "Continuing to next set of questions...")
            else:
                state["current_question_id"] = "END_MODULE_FAIL"
                state["interview_status"] = InterviewStatus.COMPLETED.value
                logger.info("CHECK_A1_A2: No symptoms detected, ending interview")
                # Add completion message
                state = add_to_transcript(state, "interviewer", "Based on your responses, you don't appear to meet the criteria for major depression at this time.")
            
            # Don't show a question, just update state
            return state
        
        elif current_id == "CHECK_DIAGNOSIS":
            # Handle diagnosis logic internally
            screener_symptoms = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
            positive_symptoms = sum(1 for q in screener_symptoms if state["symptom_map"].get(q, False))
            
            has_core_symptoms = state["symptom_map"].get("A1", False) or state["symptom_map"].get("A2", False)
            has_impairment = state["symptom_map"].get("A10", False)  
            has_exclusions = state["symptom_map"].get("A11", False)
            
            if (positive_symptoms >= 5 and has_core_symptoms and has_impairment and not has_exclusions):
                state["current_question_id"] = "END_MODULE_SUCCESS"
            else:
                state["current_question_id"] = "END_MODULE_FAIL"
                
            state["interview_status"] = InterviewStatus.COMPLETED.value
            return state
            
        # Handle endpoint states
        elif current_id.startswith("END_MODULE"):
            state["interview_status"] = InterviewStatus.COMPLETED.value
            if current_id == "END_MODULE_SUCCESS":
                message = "Thank you for completing the screening. Based on your responses, I will now provide an assessment."
            else:  # END_MODULE_FAIL
                message = "Based on your responses, you don't appear to meet the criteria for major depression at this time."
                
            state = add_to_transcript(state, "interviewer", message)
            return state
        
        # Get regular questions from the data
        questions = question_data.get("questions", question_data)
        question_info = questions.get(current_id)
        if not question_info:
            logger.error(f"Question ID {current_id} not found in question data")
            message = "I'm sorry, there seems to be an issue with the interview flow."
            state = add_safety_flag(state, f"Unknown question ID: {current_id}")
        else:
            message = question_info["question_text"]
            
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

def parse_response_node(state: AgentState, llm=None) -> str:
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
        
        # Use passed LLM or try to create one
        use_llm_parsing = True
        if llm is None:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4", temperature=0)
                logger.info("Created LLM instance for parsing")
            except Exception as e:
                logger.warning(f"Could not initialize LLM for parsing: {e}, using fallback method")
                use_llm_parsing = False
        else:
            logger.info("Using passed LLM instance for parsing")
        
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
        
        # Try LLM-based parsing first if available
        if use_llm_parsing and llm is not None:
            try:
                # Make the API call with function calling
                response = llm.bind_functions([classification_function]).invoke(messages)
                
                # Extract the classification from function call
                if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                    function_call = response.additional_kwargs['function_call']
                    if function_call['name'] == 'classify_response':
                        result = json.loads(function_call['arguments'])
                        classification = result['classification']
                        confidence = result.get('confidence', 0.5)
                        
                        logger.info(f"LLM classified response '{latest_response[:30]}...' as '{classification}' (confidence: {confidence})")
                        return classification
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, falling back to keyword matching")
        
        # Fallback parsing if function calling fails
        response_lower = latest_response.lower()
        
        # More comprehensive keyword matching for medical responses
        yes_indicators = [
            'yes', 'yeah', 'yep', 'sure', 'definitely', 'absolutely', 'certainly',
            'sometimes', 'often', 'frequently', 'usually', 'mostly', 'kind of',
            'sort of', 'i guess', 'i think so', 'pretty much', 'more or less',
            'heavy feeling', 'dark cloud', 'feel', 'been', 'having', 'experiencing',
            'trouble', 'hard', 'difficult', 'rough', 'bad', 'worse', 'tired',
            'empty', 'flat', 'grey', 'pointless', 'chore', 'burden', 'worthless'
        ]
        
        no_indicators = [
            'no', 'not', 'never', 'rarely', 'hardly', 'barely', 'nothing',
            'none', 'neither', 'nope', 'nah', 'negative', 'false', 'incorrect',
            "don't", "doesn't", "haven't", "hasn't", "can't", "won't", "isn't"
        ]
        
        # Check for strong yes indicators
        yes_count = sum(1 for word in yes_indicators if word in response_lower)
        no_count = sum(1 for word in no_indicators if word in response_lower)
        
        # If patient describes symptoms, feelings, or experiences = YES
        symptom_phrases = [
            'i have', 'i feel', 'i been', "i've been", 'it feels', 'feels like',
            'going on', 'for a while', 'past', 'weeks', 'month', 'longer',
            'everything feels', 'haven\'t been able', 'pointless', 'used to love'
        ]
        
        if any(phrase in response_lower for phrase in symptom_phrases):
            logger.info(f"Detected symptom description, classifying as 'yes': {latest_response[:50]}...")
            return "yes"
            
        if yes_count > no_count:
            logger.info(f"Response classified as 'yes' based on indicators (yes:{yes_count}, no:{no_count})")
            return "yes"
        elif no_count > yes_count:
            logger.info(f"Response classified as 'no' based on indicators (yes:{yes_count}, no:{no_count})")
            return "no"
        else:
            # For medical screening, err on side of caution
            logger.warning(f"Ambiguous response ({yes_count} yes, {no_count} no indicators), defaulting to 'yes' for safety")
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
        questions = question_data.get("questions", question_data)  # Handle both JSON structures
        if current_id in questions:
            question_info = questions[current_id]
            if question_info.get("is_screener", False):
                answer = parsed_response == "yes"
                logger.info(f"Updating symptom map: {current_id} = {answer} (response: '{parsed_response}')")
                state = update_symptom_map(state, current_id, answer)
                logger.info(f"Symptom map after update: {state['symptom_map']}")
            else:
                logger.info(f"Skipping symptom map update for non-screener question: {current_id}")
        
        # Handle regular question routing
        if current_id in questions:
            question_info = questions[current_id]
            skip_logic = question_info.get("skip_logic", {})
            
            if parsed_response == "yes":
                next_question = skip_logic.get("yes_destination")
            else:
                next_question = skip_logic.get("no_destination")
                
            if not next_question:
                logger.error(f"No destination found for {current_id} with response {parsed_response}")
                next_question = "END_MODULE_FAIL"
                
            state["current_question_id"] = next_question
            logger.info(f"Routed from {current_id} to {next_question} based on '{parsed_response}' response")
            logger.info(f"Updated symptom_map: {state['symptom_map']}")
            
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
        ("system", """You are simulating a patient in a medical interview. Answer based on your persona.

Your persona: {persona}

CRITICAL: Answer honestly based on what your persona describes!

Guidelines:
- If your persona mentions you HAVE a symptom → acknowledge it when asked directly
- If your persona mentions you DON'T have a symptom → clearly deny it
- Give realistic, brief answers (1-3 sentences)
- Speak like a real person, not a textbook

Examples for common symptoms:
- Sadness: "Yes, I've been feeling really down and heavy inside"
- Loss of interest: "Yeah, things I used to enjoy just feel pointless now"  
- Sleep problems: "Yes, I toss and turn all night, can't get good sleep"
- Feeling worthless: "Yes, I feel like I'm just a burden to everyone"
- Thoughts of death: "Yes, I've had thoughts about death, but no specific plans"

BE HONEST about your persona's symptoms when asked directly."""),
        ("human", "Interviewer: {question}\n\nRespond as the patient (keep it brief):")
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

def create_safety_monitor_agent(llm) -> Runnable:
    """
    Create a real-time safety monitoring agent that flags dangerous content.
    
    Args:
        llm: Language model instance
        
    Returns:
        Runnable safety monitor agent
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    safety_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SAFETY MONITOR for a medical interview. Your job is to immediately flag any safety concerns.

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

Be VERY sensitive to safety language even if subtle."""),
        ("human", """Patient just said: "{patient_response}"

In context of question: "{interviewer_question}"

SAFETY ASSESSMENT:""")
    ])
    
    return safety_prompt | llm | StrOutputParser()

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
    if current_id in ["END_MODULE_SUCCESS", "END_MODULE_FAIL"]:
        return True
        
    # Check if interview status indicates completion
    if state["interview_status"] in [InterviewStatus.COMPLETED.value, InterviewStatus.ESCALATED.value]:
        return True
        
    return False

def generate_diagnosis(state: AgentState) -> str:
    """
    Generate a simple patient summary based on the collected symptoms.
    
    Args:
        state: Current AgentState with symptom data
        
    Returns:
        Simple patient summary string
    """
    try:
        symptom_map = state["symptom_map"]
        positive_symptoms = [k for k, v in symptom_map.items() if v]
        total_symptoms = len(positive_symptoms)
        
        logger.info(f"=== PATIENT SUMMARY GENERATION ===")
        logger.info(f"Full symptom_map: {symptom_map}")
        logger.info(f"Positive symptoms: {positive_symptoms}")
        logger.info(f"Total positive symptoms: {total_symptoms}")
        
        # Create a simple, patient-friendly summary
        if total_symptoms == 0:
            summary = "Patient reports feeling generally well with no significant symptoms of depression."
        elif total_symptoms <= 2:
            summary = "Patient reports some mild symptoms but overall functioning well."
        elif total_symptoms <= 4:
            summary = "Patient reports several symptoms that may indicate mild to moderate depression."
        elif total_symptoms <= 6:
            summary = "Patient reports multiple symptoms consistent with moderate depression."
        else:
            summary = "Patient reports numerous symptoms that may indicate significant depression."
        
        # Add specific symptom mentions if any
        if symptom_map.get("A9", False):  # Suicidal ideation
            summary += " Patient has reported thoughts of death or self-harm."
        
        logger.info(f"Patient summary: {summary}")
        logger.info(f"=== END PATIENT SUMMARY ===")
        
        return summary
            
    except Exception as e:
        logger.error(f"Error generating patient summary: {e}")
        return "Unable to generate patient summary due to technical error" 