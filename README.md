# AI Research Team - LLM Training Efficiency Lab

An autonomous AI research team powered by Lumenbridge that discovers more efficient ways to train LLMs.

## 🎯 Overview

This system creates a collaborative team of 4 specialized AI agents that work together in 15-minute sprint cycles to:

- Research novel LLM training techniques
- Design and test model architectures
- Implement experiments in Python
- Validate results and iterate

## 🤖 Team Structure

Based on Lumenbridge's recommendation, our team consists of:

### 1. **ResearcherAgent** 🔬

- **Role**: AI researcher
- **Expertise**: LLM architectures, training efficiency, optimization algorithms
- **Responsibilities**: Generate hypotheses, analyze tradeoffs, propose experiments

### 2. **ArchitectAgent** 🏗️

- **Role**: System architect
- **Expertise**: Model design, training pipelines, system optimization
- **Responsibilities**: Translate research into concrete designs and experimental plans

### 3. **CoderAgent** 💻

- **Role**: Python developer
- **Expertise**: PyTorch/JAX, efficient ML code, dependency management
- **Responsibilities**: Implement experiments with production-quality code

### 4. **ValidatorAgent** ✅

- **Role**: ML validation expert
- **Expertise**: Experimental validation, benchmarking, result analysis
- **Responsibilities**: Validate correctness, interpret results, ensure rigor

## 🏃 Sprint Cycle (15 minutes)

Each sprint follows this structure:

### Phase 1: Discussion (5 min) 📋

- All agents contribute their perspectives
- Identify challenges and opportunities
- Synthesize into unified action plan

### Phase 2: Implementation (8 min) 💻

- CoderAgent generates Python code
- Code is executed in sandbox environment
- Results captured automatically

### Phase 3: Validation (2 min) ✅

- ValidatorAgent analyzes results
- Scientific validity assessment
- Recommendations for next sprint

## 🚀 Getting Started

### Prerequisites

```bash
pip install requests python-dotenv
```

### Setup

1. Configure `.env`:

```bash
SUDO_PASSWORD="your_password"
LUMENBRIDGE_BASE_URL="https://lumenbridge.xyz"
AI_RESEARCH_USER_ID="ai-research-lab"
```

2. Run the system:

```bash
python ai_research_team.py
```

## 📁 Project Structure

```
ai-gone-crazy/
├── .env                          # Environment configuration
├── ai_research_team.py           # Main orchestrator
├── consult_lumenbridge.py        # API consultation tool
├── sandbox/                      # Agent workspace
└── results/                      # Sprint outputs
    └── sprint_YYYYMMDD_HHMMSS/
        ├── sprint_summary.json   # Complete sprint data
        ├── experiment.py         # Generated code
        └── requirements.txt      # Dependencies
```

## 🔧 How It Works

### 1. Team Registration

```python
team = AIResearchTeam()
team.setup_team()  # Registers 4 agents with Lumenbridge
```

### 2. Run Sprint

```python
research_goal = "Test learning rate warmup strategies..."
sprint_result = team.run_sprint(research_goal)
```

### 3. Behind the Scenes

- **Discussion**: Each agent invoked via `/api/agents/invoke-user-agent`
- **Code Generation**: Uses `/api/agents/code` endpoint
- **Execution**: Runs Python in sandboxed subprocess
- **Validation**: Agent analyzes outputs for insights

## 🎯 Example Research Goals

Try these research questions:

```python
# Learning rate optimization
"Compare AdamW vs. Lion optimizer on small transformer training"

# Architecture experiments
"Test if reducing hidden dimension by 50% impacts final loss"

# Training efficiency
"Measure memory savings from gradient checkpointing"

# Numerical stability
"Test mixed precision (fp16) vs. full precision training stability"
```

## 📊 Sprint Output Format

Each sprint produces:

```json
{
  "sprint_id": "20251203_120000",
  "goal": "Research question...",
  "phases": {
    "discussion": {
      "individual_responses": {...},
      "action_plan": "..."
    },
    "implementation": {
      "code": "...",
      "execution": {...}
    },
    "validation": {
      "analysis": "...",
      "execution_success": true
    }
  }
}
```

## 🔐 Capabilities

### Python Execution

- ✅ Create virtual environments
- ✅ Install libraries via pip
- ✅ Execute code with timeout (2 min max)
- ✅ Capture stdout/stderr
- ✅ Save results automatically

### Safety Features

- 🛡️ Sandboxed execution directory
- ⏱️ Timeout protection (120s)
- 📝 Full logging and error capture
- 🔄 Clean failure recovery

## 🌉 Lumenbridge Integration

This system uses these Lumenbridge endpoints:

| Endpoint                        | Purpose                      |
| ------------------------------- | ---------------------------- |
| `/api/agents/register`          | Register custom agents       |
| `/api/agents/invoke-user-agent` | Execute agent with context   |
| `/api/agents/code`              | Generate Python code         |
| `/api/router`                   | Route requests to best agent |

### Authentication

- Currently using public API (no auth required)
- Can add BSV key signing for enhanced security

## 🚀 Advanced Usage

### Custom Research Goals

```python
team = AIResearchTeam()
team.setup_team()

# Run multiple sprints
goals = [
    "Test gradient clipping thresholds",
    "Compare batch sizes for training stability",
    "Evaluate weight decay impact on generalization"
]

for goal in goals:
    team.run_sprint(goal)
```

### Analyzing Results

```python
# All sprints logged
for sprint in team.sprint_log:
    print(f"Sprint {sprint['sprint_id']}: {sprint['goal']}")
    if sprint['phases']['validation']['execution_success']:
        print("✅ Success!")
```

## 🤝 Contributing

This is an experimental research platform. Ideas for improvement:

- [ ] Add persistent memory across sprints
- [ ] Implement learning from past experiments
- [ ] Add visualization of training curves
- [ ] Integrate with Weights & Biases
- [ ] Multi-GPU experiment support
- [ ] Automated hyperparameter search

## 📚 Learn More

- **Lumenbridge API**: https://lumenbridge.xyz/api-doc
- **Schema Reference**: `lumenbridge_team_recommendation.json`
- **Agent Prompts**: See `ai_research_team.py` setup_team()

## ⚡ Quick Test

Run a quick test sprint:

```bash
python ai_research_team.py
```

This will:

1. Register the 4-agent team
2. Run a learning rate warmup experiment
3. Save results to `results/sprint_*/`
4. Display full analysis

---

**Built with 🌉 Lumenbridge - Self-aware AI agents that build the future**

_Last Updated: December 3, 2025_
