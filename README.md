# 🏥 Three-Agent Medical Interview System

A sophisticated AI-powered medical interview simulation system that uses **LangGraph** and **LangChain** to orchestrate three specialized agents for conducting, simulating, and evaluating medical interviews.

## 🏗️ Architecture Overview

This system implements the architecture you specified:

```mermaid
graph TD
    subgraph "User's Device (Browser/React UI)"
        A[User Sees Question in UI] --> B[User Types Answer in Text Box];
        B --> C{User Clicks 'Send'};
        C --> D[Send Text Data to Backend];
    end

    subgraph "Backend Server (Python/FastAPI)"
        D --> E[1. Receive User Text];
        E --> F{2. Process with Agents};

        subgraph "Agentic Core (LangChain)"
            F -- User Text --> G(Interviewer Agent);
            F -- User Text --> H(Judge Agent - Parallel);

            G -- Decides Next Step --> I[MINI Logic (.json)];
            G -- Updates State --> J[Session State];
            G -- Generates Response --> K(OpenAI GPT API);

            H -- Scans for Risk & Logic Errors --> L[Safety & QA Check];
            L -- Risk Detected? --> M{Trigger Safety Protocol};
        end

        K -- Next Question Text --> N[3. Send New Question Text to UI];
        N --> A;

        M -- Yes --> O[Log & Escalate to Human];
        G -- Interview Complete? --> P[4. Log Results];
    end

    subgraph "Database (e.g., MongoDB)"
        P --> Q[Save Final Report];
    end
```

## 🤖 Three Agent System

### 1. **Interviewer Agent** (LangGraph-based)
- **`get_question_node`**: Reads `mini_module_a.json` and presents questions
- **`parse_response_node`**: Uses OpenAI function calling to classify responses as yes/no
- **`route_logic_node`**: Implements complex routing logic including special CHECK_A1_A2 flow

### 2. **Patient Agent** (LangChain Runnable)  
- Simulates realistic patient responses based on predefined personas
- Generates natural, human-like conversation
- Maintains consistency with patient background

### 3. **Judge Agent** (LangChain Runnable)
- Evaluates interviewer performance against ground truth
- Scores accuracy, completeness, safety, flow, and communication
- Provides detailed feedback and improvement suggestions

## 📁 Project Structure

```
AGenticAI/
├── app.py                  # 🌐 Streamlit Web UI (Main Interface)
├── run_ui.py               # 🚀 Web UI launcher script
├── setup_guide.py          # 📋 Interactive setup guide
├── install_gemini.py       # 🔧 Quick Gemini installer
├── main.py                 # 🖥️ Command-line simulation orchestrator
├── agents.py               # 🤖 Three agent definitions
├── utils.py                # 🛠️ AgentState and utility functions  
├── mini_module_a.json      # 📋 Interview question flow logic
├── requirements.txt        # 📦 Python dependencies
└── README.md              # 📚 This documentation
```

## 🚀 Quick Start

### 0. First Time? Run the Setup Guide! ⭐

```bash
# Get step-by-step instructions
python setup_guide.py
```

This interactive guide will check your setup and show you exactly how to get your OpenAI API key!

### 1. Setup Environment

```bash
# Clone/navigate to project directory
cd AGenticAI

# Install dependencies
pip install -r requirements.txt

# Optional: Add Gemini support
python install_gemini.py
```

### 2. Option A: Web UI (Recommended) 🌐

```bash
# Launch the beautiful web interface
python run_ui.py

# Or directly with streamlit
streamlit run app.py
```

**Features:**
- 🏗️ **Architecture Visualization** - See the system diagram
- 🤖 **Multiple AI Providers** - Choose OpenAI or Google Gemini
- 🔑 **Easy API Key Setup** - Enter your API key in the sidebar  
- 💬 **Real-time Conversation** - Watch the interview unfold
- 📊 **Live Status Monitoring** - Track progress and risk levels
- ⚖️ **Judge Evaluation** - Get detailed performance analysis
- 💾 **Session Management** - Save and load interviews

The web app will open at `http://localhost:8501`

### 3. Option B: Command Line Interface

```bash
# Set API key (choose one)
export OPENAI_API_KEY="your-openai-key"     # For OpenAI
export GOOGLE_API_KEY="your-gemini-key"     # For Google Gemini

# Run console simulation
python main.py
```

This will run 4 different patient scenarios and show you:
- 🩺 Real-time conversation flow
- 📊 Interview completion status
- ⚖️ Judge evaluation and scoring
- 📈 Summary statistics

### 3. Example Output

```
🏥 MEDICAL INTERVIEW SIMULATION - Session: a1b2c3d4
================================================================================
📋 Patient Persona: A 45-year-old man who has been feeling profoundly sad...
🎯 Ground Truth: Meets criteria for Major Depressive Episode
================================================================================

[14:23:15] Turn 1 - 🩺 INTERVIEWER:
  To begin, I will ask you about major depressive episodes. In the last two weeks, have you been feeling sad, empty, or depressed most of the day, nearly every day?

[14:23:16] Turn 2 - 👤 PATIENT:
  Yes, I have been feeling very sad and empty most days. It's been going on for about a month now, and it's much worse than my usual mood.

...

🏁 INTERVIEW COMPLETED
============================================================
📊 Final Status: completed
⚠️  Risk Level: high
🩺 Diagnosis: Likely Major Depressive Episode - Recommend clinical evaluation
📈 Questions Answered: 9
✅ Positive Symptoms: 6

⚖️  JUDGE EVALUATION
============================================================
## Overall Performance: Good

## Detailed Analysis:
- **Accuracy**: 8/10 - Correctly identified major depressive episode
- **Completeness**: 9/10 - Asked all relevant screening questions
- **Safety**: 7/10 - Appropriately flagged suicide risk
- **Flow**: 8/10 - Logical question sequencing
- **Communication**: 9/10 - Clear, professional questions

## Overall Score: 8.2/10
```

## 🔧 Customization

### Adding New Patient Personas

Edit the `load_patient_personas()` function in `main.py`:

```python
personas = [
    {
        'persona': 'Your patient description here...',
        'ground_truth': 'Expected diagnosis here...'
    }
]
```

### Modifying Interview Questions

Edit `mini_module_a.json` to change:
- Question text and flow
- Routing logic
- Special conditions
- Diagnostic criteria

### Adjusting Agent Behavior

- **Interviewer**: Modify functions in `agents.py`
- **Patient**: Update prompts in `create_patient_agent()`
- **Judge**: Update evaluation criteria in `create_judge_agent()`

## 📊 Key Features

### ✅ **Implemented from Your Specifications**
- [x] LangGraph AgentState with all required fields
- [x] Three LangGraph nodes for interviewer logic
- [x] OpenAI function calling for response parsing
- [x] Special CHECK_A1_A2 routing logic
- [x] Patient and Judge LangChain runnables
- [x] Complete simulation loop
- [x] Final diagnosis generation
- [x] Judge evaluation system

### 🚀 **Enhanced for Production**
- [x] **🌐 Beautiful Web UI** with Streamlit interface
- [x] **🤖 Multi-Provider Support** - OpenAI & Google Gemini
- [x] **🏗️ Architecture Visualization** with interactive diagrams
- [x] **🔑 Easy API Key Management** in the sidebar
- [x] **💬 Real-time Conversation Display** with color-coded messages
- [x] **📊 Live Status Monitoring** and risk assessment
- [x] **⚖️ Interactive Judge Evaluation** with expandable details
- [x] **💾 Session Management** with save/load functionality
- [x] Comprehensive error handling
- [x] Multiple patient scenarios
- [x] JSON session data export

## 🔒 Safety Features

- **Risk Level Monitoring**: Tracks LOW → MEDIUM → HIGH → CRITICAL
- **Safety Flags**: Automatically flags concerning responses
- **Suicide Risk Detection**: Special handling for A9 responses
- **Emergency Protocols**: Built-in escalation pathways
- **Error Recovery**: Graceful degradation on failures

## 🎯 Use Cases

- **Medical Training**: Train healthcare workers on interview techniques
- **AI Testing**: Validate conversational AI systems for healthcare
- **Research**: Study interview flow optimization
- **Quality Assurance**: Automated evaluation of medical interviews
- **Simulation**: Safe environment for testing sensitive scenarios

## 📈 Scalability

The system is designed for easy expansion:

- **New Modules**: Add additional JSON files for different conditions
- **Multiple Languages**: Extend prompts for international use
- **Database Integration**: Replace file storage with proper databases
- **Web Interface**: Add FastAPI/React frontend per your original diagram
- **Parallel Processing**: Run multiple interviews simultaneously

## 🛠️ Development

To extend the system:

1. **Add new agent types** in `agents.py`
2. **Extend AgentState** in `utils.py` for new fields
3. **Create new interview modules** as JSON files
4. **Add evaluation metrics** in judge agent prompts
5. **Integrate with databases** using `serialize_state_for_db()`

## 🤝 Contributing

This system implements your exact specifications and is ready for production enhancement. The modular design makes it easy to add new features while maintaining the core three-agent architecture.

---

**🎉 Your three-agent medical interview system is ready to run!** 