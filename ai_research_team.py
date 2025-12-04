"""
AI Research Team Orchestrator
Based on Lumenbridge's recommendation for LLM training efficiency research

Team Structure:
- Researcher: Literature review, hypothesis generation
- Architect: System design, model architecture planning  
- Coder: Python implementation
- Validator: Testing, benchmarking, result analysis

Sprint Cycle: 15 minutes
- Phase 1 (5 min): Collaborative Discussion
- Phase 2 (8 min): Implementation
- Phase 3 (2 min): Validation & Analysis
"""

import requests
import json
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "https://lumenbridge.xyz"
USER_ID = "ai-research-lab"
SANDBOX_DIR = Path("./sandbox")
RESULTS_DIR = Path("./results")
SUDO_PASSWORD = os.getenv("SUDO_PASSWORD", "")

# Ensure directories exist
SANDBOX_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class LumenbridgeClient:
    """Client for interacting with Lumenbridge API"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def code_generator(self, prompt, context=None):
        """Call CodeGenerator agent"""
        url = f"{self.base_url}/api/agents/code"
        payload = {"prompt": prompt}
        if context:
            payload["context"] = context
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def terminal_agent(self, task, context=None):
        """Call TerminalAgent"""
        url = f"{self.base_url}/api/agents/terminal"
        payload = {"task": task}
        if context:
            payload["context"] = context
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def schema_agent(self, user_prompt):
        """Call SchemaAgent"""
        url = f"{self.base_url}/api/agents/schema"
        payload = {"userPrompt": user_prompt}
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def register_user_agent(self, name, description, prompt, metadata=None):
        """Register a custom user agent"""
        url = f"{self.base_url}/api/agents/register"
        payload = {
            "userId": USER_ID,
            "name": name,
            "description": description,
            "prompt": prompt
        }
        if metadata:
            payload["metadata"] = metadata
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def invoke_user_agent(self, agent_name, context):
        """Invoke a registered user agent"""
        url = f"{self.base_url}/api/agents/invoke-user-agent"
        payload = {
            "userId": USER_ID,
            "agentName": agent_name,
            "context": context
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_my_agents(self):
        """Get all registered agents for this user"""
        url = f"{self.base_url}/api/agents/my-agents/{USER_ID}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


class AIResearchTeam:
    """Orchestrates AI research team for LLM optimization"""
    
    def __init__(self):
        self.client = LumenbridgeClient()
        self.sprint_log = []
        
    def setup_team(self):
        """Register the 4 specialist agents"""
        print("\n🚀 Setting up AI Research Team...\n")
        
        agents = [
            {
                "name": "ResearcherAgent",
                "description": "AI researcher specializing in LLM training efficiency, novel architectures, and optimization techniques",
                "prompt": """You are an expert AI researcher with deep knowledge of:
- Large Language Model architectures (Transformers, attention mechanisms, positional encodings)
- Training efficiency techniques (mixed precision, gradient checkpointing, activation checkpointing)
- Optimization algorithms (AdamW, LAMB, Adafactor)
- Novel research papers on model compression, quantization, pruning
- Distributed training strategies (data parallelism, model parallelism, pipeline parallelism)

Your role: Generate research hypotheses, analyze tradeoffs, propose experiments based on cutting-edge research.
Always provide scientific reasoning and cite concepts from recent literature.""",
                "metadata": {"role": "researcher", "specialty": "llm-optimization"}
            },
            {
                "name": "ArchitectAgent",
                "description": "AI architect designing efficient model architectures and training pipelines",
                "prompt": """You are a senior AI architect specializing in:
- Model architecture design (layer configurations, attention patterns, feed-forward networks)
- Training pipeline optimization (data loading, batching strategies, memory management)
- System-level design decisions (hardware utilization, distributed training setups)
- Experiment design (control variables, baselines, metrics)

Your role: Translate research ideas into concrete architectures and experimental designs.
Provide detailed system specifications, flowcharts (in text), and implementation plans.
Consider practical constraints like memory, compute, and time.""",
                "metadata": {"role": "architect", "specialty": "system-design"}
            },
            {
                "name": "CoderAgent",
                "description": "Python expert implementing ML experiments with PyTorch, JAX, and efficient training code",
                "prompt": """You are an expert Python developer specializing in:
- PyTorch and JAX for deep learning
- Efficient tensor operations and memory management
- Python virtual environments, dependency management
- Writing clean, documented, production-quality code
- Debugging and error handling
- Energy-efficient ML implementations
- Model quantization, pruning, and compression
- Real experiment tracking with metrics

Your role: Implement REAL, RUNNABLE experiments that produce actual results.
Generate complete code with:
- Proper imports (torch, numpy, time, json)
- Actual model training (not just structure)
- Comprehensive metrics (loss, accuracy, time, memory)
- Energy tracking when possible
- Results saved to JSON files
- Clear output with print statements

Use modern Python (3.10+). Code must execute quickly (<2 min).
Focus on energy-efficient training methods and model compression.""",
                "metadata": {"role": "coder", "specialty": "ml-implementation"}
            },
            {
                "name": "ValidatorAgent",
                "description": "ML engineer validating experiments, analyzing results, and ensuring scientific rigor",
                "prompt": """You are an ML validation expert specializing in:
- Experimental validation (statistical significance, reproducibility)
- Performance analysis (training curves, loss landscapes, gradient norms)
- Benchmarking against baselines
- Result interpretation and insights extraction
- Identifying bugs, numerical instabilities, and implementation errors

Your role: Analyze experimental outputs, validate correctness, interpret results.
Provide critical feedback, identify issues, suggest improvements.
Ensure scientific rigor and reproducibility.""",
                "metadata": {"role": "validator", "specialty": "validation"}
            },
            {
                "name": "DebuggerAgent",
                "description": "Expert debugger that analyzes errors and provides self-healing solutions",
                "prompt": """You are an expert debugging specialist with deep knowledge of:
- Python error analysis (stack traces, exceptions, module errors)
- PyTorch/ML framework common issues (CUDA, tensor types, memory, device placement)
- Environment issues (missing packages, version conflicts, path problems)
- Code fix strategies (type conversions, imports, error handling)
- Root cause analysis and prevention

Your role: Analyze errors from experiment execution and provide ACTIONABLE fixes.

For each error, provide:
1. **Root Cause**: What actually caused the error (be specific)
2. **Immediate Fix**: Exact command(s) or code changes needed
3. **Prevention**: How to avoid this in the future
4. **Confidence**: High/Medium/Low on fix success

Common error patterns:
- ModuleNotFoundError → pip install package
- Tensor type errors → .long(), .float(), .to(device)
- Shape mismatches → verify dimensions, add reshaping
- CUDA errors → device placement, memory management
- Import errors → check file paths, add to sys.path

Provide terminal commands in PowerShell format when applicable.
Be concise and actionable - we need to fix and continue quickly.""",
                "metadata": {"role": "debugger", "specialty": "error-recovery"}
            }
        ]
        
        registered = []
        for agent_config in agents:
            try:
                result = self.client.register_user_agent(**agent_config)
                print(f"✅ Registered: {agent_config['name']}")
                registered.append(agent_config['name'])
            except requests.exceptions.HTTPError as e:
                error_text = str(e.response.text if hasattr(e, 'response') and e.response else e)
                if "Duplicate agent name" in error_text or "already exists" in error_text or "409" in str(e):
                    print(f"⚠️  {agent_config['name']} already exists (skipping)")
                    registered.append(agent_config['name'])
                else:
                    print(f"❌ Failed to register {agent_config['name']}: {e}")
                    print(f"   Error details: {error_text[:200]}")
        
        print(f"\n✅ Team setup complete! {len(registered)}/5 agents ready.\n")
        return registered
    
    def run_sprint(self, research_goal):
        """Execute a 15-minute sprint cycle"""
        sprint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        sprint_dir = RESULTS_DIR / f"sprint_{sprint_id}"
        sprint_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🏃 SPRINT {sprint_id}: {research_goal}")
        print(f"{'='*80}\n")
        
        sprint_data = {
            "sprint_id": sprint_id,
            "goal": research_goal,
            "start_time": datetime.now().isoformat(),
            "phases": {}
        }
        
        # Phase 1: Collaborative Discussion (5 minutes)
        print("\n📋 PHASE 1: Collaborative Discussion (5 min)")
        print("-" * 80)
        discussion = self._phase_discussion(research_goal)
        sprint_data["phases"]["discussion"] = discussion
        
        # Phase 2: Implementation (8 minutes)
        print("\n💻 PHASE 2: Implementation (8 min)")
        print("-" * 80)
        implementation = self._phase_implementation(discussion, sprint_dir)
        sprint_data["phases"]["implementation"] = implementation
        
        # Self-healing: If implementation failed, attempt recovery
        if not implementation.get("execution", {}).get("success", False):
            print("\n🔧 SELF-HEALING: Attempting error recovery...")
            print("-" * 80)
            healing_result = self._attempt_self_healing(implementation, discussion, sprint_dir)
            sprint_data["phases"]["self_healing"] = healing_result
            
            # If healing provided a fix, update implementation
            if healing_result.get("fix_applied", False):
                implementation = healing_result.get("new_implementation", implementation)
                sprint_data["phases"]["implementation"] = implementation
        
        # Phase 3: Validation (2 minutes)
        print("\n✅ PHASE 3: Validation & Analysis (2 min)")
        print("-" * 80)
        validation = self._phase_validation(implementation, sprint_dir)
        sprint_data["phases"]["validation"] = validation
        
        sprint_data["end_time"] = datetime.now().isoformat()
        
        # Save sprint results
        sprint_file = sprint_dir / "sprint_summary.json"
        with open(sprint_file, 'w') as f:
            json.dump(sprint_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ SPRINT COMPLETE!")
        print(f"📁 Results saved to: {sprint_dir}")
        print(f"{'='*80}\n")
        
        self.sprint_log.append(sprint_data)
        return sprint_data
    
    def _phase_discussion(self, research_goal):
        """Phase 1: Agents discuss and plan"""
        discussion_context = {
            "userPrompt": f"""Research Goal: {research_goal}

As the team's {"{role}"}, provide your perspective:
1. What are the key challenges and opportunities?
2. What approach would you recommend?
3. What are the potential risks and mitigations?
4. What would success look like?

Be specific and actionable."""
        }
        
        agents = ["ResearcherAgent", "ArchitectAgent", "CoderAgent", "ValidatorAgent"]
        responses = {}
        
        for agent in agents:
            context = {
                "userPrompt": discussion_context["userPrompt"].format(
                    role=agent.replace("Agent", "")
                )
            }
            
            print(f"\n🤖 {agent} contributing...")
            try:
                result = self.client.invoke_user_agent(agent, context)
                response_text = result.get("result", {}).get("response", "")
                responses[agent] = response_text
                print(f"   ✅ Response: {response_text[:200]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                responses[agent] = f"Error: {e}"
        
        # Synthesize discussion into a plan
        print("\n🧠 Synthesizing discussion into action plan...")
        synthesis_prompt = f"""Based on team discussion:

RESEARCHER: {responses.get('ResearcherAgent', 'N/A')[:500]}

ARCHITECT: {responses.get('ArchitectAgent', 'N/A')[:500]}

CODER: {responses.get('CoderAgent', 'N/A')[:500]}

VALIDATOR: {responses.get('ValidatorAgent', 'N/A')[:500]}

Create a unified action plan:
1. Hypothesis to test
2. Experimental approach
3. Implementation steps
4. Success criteria
5. Key metrics to track

Be concise and implementable in 8 minutes."""
        
        try:
            plan_result = self.client.invoke_user_agent(
                "ArchitectAgent",
                {"userPrompt": synthesis_prompt}
            )
            action_plan = plan_result.get("result", {}).get("response", "")
            print(f"\n📝 Action Plan:\n{action_plan}\n")
        except Exception as e:
            print(f"❌ Plan synthesis failed: {e}")
            action_plan = "Plan synthesis failed - proceeding with individual inputs"
        
        return {
            "individual_responses": responses,
            "action_plan": action_plan
        }
    
    def _phase_implementation(self, discussion, sprint_dir):
        """Phase 2: Generate and execute code"""
        action_plan = discussion.get("action_plan", "")
        
        # Generate implementation code
        print("\n💻 CoderAgent generating implementation...")
        code_prompt = f"""Based on this action plan:

{action_plan}

Generate a COMPLETE, EXECUTABLE Python script for a REAL ML experiment.

AVAILABLE UTILITIES (you can import these):
```python
from ml_utils import (
    create_synthetic_mnist,      # Returns DataLoader with correct types
    get_model_size,               # Calculate model parameters and size
    measure_inference_time,       # Measure inference speed
    train_simple_classifier,      # Standard training loop with error handling
    evaluate_accuracy,            # Evaluate model accuracy
    save_experiment_results,      # Save metrics to JSON
    SimpleMLP,                    # Simple 2-layer MLP (784→128→10)
    DeepMLP                       # 4-layer MLP (784→512→256→128→10)
)
```

YOUR SCRIPT MUST:
1. Import required libraries (torch, numpy, time, json, ml_utils)
2. Define models OR use SimpleMLP/DeepMLP from ml_utils
3. Create training data using create_synthetic_mnist() or custom synthetic data
4. Implement training with proper PyTorch types:
   - Use .long() for classification labels
   - Use .float() for input tensors
   - Verify tensor shapes before loss calculation
5. Track comprehensive metrics:
   - Training time
   - Final loss/accuracy
   - Model size (use get_model_size)
   - Inference time (use measure_inference_time)
6. Print clear progress every few steps
7. Save results to JSON at the end

CRITICAL PYTORCH REQUIREMENTS:
- Always use .long() or .to(torch.long) for classification labels
- Use .float() for input tensors
- Test tensor shapes before loss calculation
- Handle device placement (CPU/CUDA) consistently
- Use torch.no_grad() for evaluation/inference

ERROR PREVENTION CHECKLIST:
✓ Labels are torch.long (not int32)
✓ Model output shape matches loss function expectations
✓ All tensors on same device
✓ Batch dimensions aligned
✓ Data loader yields correct types

The code should complete in under 3 minutes.
Focus on energy-efficient methods when applicable.

Example structure:
```python
import torch
from ml_utils import create_synthetic_mnist, SimpleMLP, train_simple_classifier, get_model_size

# Create data
train_loader = create_synthetic_mnist(num_samples=1000, batch_size=32)

# Create model
model = SimpleMLP()

# Train
metrics = train_simple_classifier(model, train_loader, num_epochs=5)

# Analyze
print(f"Model size: {{get_model_size(model)}}")
print(f"Training time: {{metrics['training_time']}}s")
```
"""
        
        try:
            code_result = self.client.code_generator(
                prompt=code_prompt,
                context={"language": "python", "includeTests": False}
            )
            
            code = code_result.get("result", {}).get("code", "")
            explanation = code_result.get("result", {}).get("explanation", "")
            dependencies = code_result.get("result", {}).get("dependencies", [])
            
            print(f"\n📝 Code generated ({len(code)} chars)")
            print(f"📦 Dependencies: {', '.join(dependencies) if dependencies else 'None'}")
            print(f"💡 Explanation: {explanation[:200]}...\n")
            
            # Save code
            code_file = sprint_dir / "experiment.py"
            with open(code_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(code)
            
            # Copy ml_utils.py to experiment directory for imports
            import shutil
            ml_utils_src = Path(__file__).parent / "ml_utils.py"
            if ml_utils_src.exists():
                ml_utils_dst = sprint_dir / "ml_utils.py"
                shutil.copy(ml_utils_src, ml_utils_dst)
            
            # Create requirements.txt
            if dependencies:
                req_file = sprint_dir / "requirements.txt"
                with open(req_file, 'w') as f:
                    f.write('\n'.join(dependencies))
            
            # Execute code
            print("🚀 Executing experiment...")
            execution_result = self._execute_python(code_file, sprint_dir)
            
            return {
                "code": code,
                "explanation": explanation,
                "dependencies": dependencies,
                "execution": execution_result
            }
            
        except Exception as e:
            print(f"❌ Implementation failed: {e}")
            return {"error": str(e)}
    
    def _execute_python(self, code_file, sprint_dir):
        """Execute Python code in sandbox"""
        try:
            # Use virtual environment Python if available
            import sys
            python_exe = sys.executable
            
            # Run in the actual sprint directory (not nested)
            # code_file is already in sprint_dir, so just use filename
            result = subprocess.run(
                [python_exe, code_file.name],
                cwd=str(code_file.parent),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300  # 5 minute timeout for training
            )
            
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
            
            if output["success"]:
                print(f"✅ Execution successful!")
                print(f"📊 Output:\n{result.stdout[:500]}")
            else:
                print(f"❌ Execution failed (code {result.returncode})")
                print(f"⚠️  Error:\n{result.stderr[:500]}")
            
            return output
            
        except subprocess.TimeoutExpired:
            print("⏱️  Execution timed out (>2 min)")
            return {"error": "Timeout", "success": False}
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return {"error": str(e), "success": False}
    
    def _phase_validation(self, implementation, sprint_dir):
        """Phase 3: Validate and analyze results"""
        execution = implementation.get("execution", {})
        
        if not execution.get("success"):
            validation_prompt = f"""The experiment failed to execute:

Error: {execution.get('stderr', 'Unknown error')}
Code: {implementation.get('code', 'N/A')[:1000]}

Analyze what went wrong and suggest fixes."""
        else:
            validation_prompt = f"""The experiment executed successfully:

Output:
{execution.get('stdout', 'No output')}

Code:
{implementation.get('code', 'N/A')[:1000]}

Analyze the results:
1. Are the results scientifically valid?
2. What insights can we extract?
3. What are the limitations?
4. What should we try next?
5. Rate the experiment (1-10) and explain why."""
        
        print("\n🔍 ValidatorAgent analyzing results...")
        try:
            validation_result = self.client.invoke_user_agent(
                "ValidatorAgent",
                {"userPrompt": validation_prompt}
            )
            
            analysis = validation_result.get("result", {}).get("response", "")
            print(f"\n📊 Analysis:\n{analysis}\n")
            
            return {
                "analysis": analysis,
                "execution_success": execution.get("success", False)
            }
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {"error": str(e)}
    
    def _attempt_self_healing(self, failed_implementation, discussion, sprint_dir, max_attempts=2):
        """Self-healing mechanism: Analyze errors and attempt fixes"""
        healing_log = {
            "attempts": [],
            "fix_applied": False,
            "new_implementation": None
        }
        
        execution = failed_implementation.get("execution", {})
        error_output = execution.get("stderr", "") + "\n" + execution.get("stdout", "")
        original_code = failed_implementation.get("code", "")
        
        for attempt in range(max_attempts):
            print(f"\n🔍 Healing Attempt {attempt + 1}/{max_attempts}")
            
            # Step 1: DebuggerAgent analyzes the error
            debug_prompt = f"""Analyze this experiment error and provide a fix:

**Original Goal:**
{discussion.get('action_plan', '')[:500]}

**Error Output:**
{error_output[:2000]}

**Failed Code:**
```python
{original_code[:1500]}
```

Provide:
1. **Root Cause**: Exact cause of the error
2. **Fix Strategy**: What needs to change (code or environment)
3. **Terminal Commands**: If packages/env changes needed (PowerShell format)
4. **Code Fix**: If code changes needed, provide the corrected section
5. **Confidence**: High/Medium/Low

Be specific and actionable."""
            
            try:
                debug_result = self.client.invoke_user_agent("DebuggerAgent", {"userPrompt": debug_prompt})
                debug_analysis = debug_result.get("result", {}).get("response", "")
                print(f"🔍 DebuggerAgent analysis:\n{debug_analysis[:500]}...\n")
                
                # Step 2: Team consensus on the fix
                consensus_prompt = f"""DebuggerAgent identified this issue and solution:

{debug_analysis}

As the {"{role}"}, do you:
1. AGREE with this fix approach? (Yes/No/Partially)
2. Any concerns or alternative suggestions?
3. What's your confidence this will work? (High/Medium/Low)

Be brief (2-3 sentences)."""
                
                votes = {}
                agents = ["ResearcherAgent", "ArchitectAgent", "CoderAgent"]
                
                for agent in agents:
                    try:
                        vote_result = self.client.invoke_user_agent(
                            agent,
                            {"userPrompt": consensus_prompt.format(role=agent.replace("Agent", ""))}
                        )
                        votes[agent] = vote_result.get("result", {}).get("response", "")
                        print(f"   {agent}: {votes[agent][:100]}...")
                    except Exception as e:
                        print(f"   {agent}: Error - {e}")
                        votes[agent] = "Error"
                
                # Step 3: Apply fix if consensus
                agreement_count = sum(1 for v in votes.values() if "yes" in v.lower() or "agree" in v.lower())
                
                if agreement_count >= 2:  # Majority agrees
                    print(f"\n✅ Consensus reached ({agreement_count}/3 agree). Applying fix...")
                    
                    # Check if terminal commands needed
                    if "pip install" in debug_analysis.lower() or "install" in debug_analysis.lower():
                        # Extract package name
                        import re
                        packages = re.findall(r'pip install ([a-zA-Z0-9_-]+)', debug_analysis)
                        if packages:
                            print(f"📦 Installing packages: {packages}")
                            # Note: In production, we'd execute this
                            # For now, log it
                            healing_log["attempts"].append({
                                "attempt": attempt + 1,
                                "analysis": debug_analysis,
                                "votes": votes,
                                "action": f"Would install: {packages}",
                                "consensus": "yes"
                            })
                    
                    # If code fix suggested, regenerate code
                    if "code" in debug_analysis.lower() and "fix" in debug_analysis.lower():
                        print("🔨 Regenerating code with fix...")
                        
                        fix_prompt = f"""Based on this error analysis:

{debug_analysis}

Regenerate the COMPLETE experiment code with the fix applied.
Use the ml_utils when possible.
Ensure all types are correct (.long(), .float()).

{discussion.get('action_plan', '')[:500]}"""
                        
                        try:
                            code_result = self.client.code_generator(
                                prompt=fix_prompt,
                                context={"language": "python", "includeTests": False}
                            )
                            
                            new_code = code_result.get("result", {}).get("code", "")
                            
                            if new_code:
                                # Save fixed code
                                code_file = sprint_dir / "experiment_fixed.py"
                                with open(code_file, 'w', encoding='utf-8', errors='replace') as f:
                                    f.write(new_code)
                                
                                # Copy ml_utils
                                import shutil
                                ml_utils_src = Path(__file__).parent / "ml_utils.py"
                                if ml_utils_src.exists():
                                    ml_utils_dst = sprint_dir / "ml_utils.py"
                                    shutil.copy(ml_utils_src, ml_utils_dst)
                                
                                # Execute fixed code
                                print("🚀 Executing fixed code...")
                                fixed_execution = self._execute_python(code_file, sprint_dir)
                                
                                if fixed_execution.get("success", False):
                                    print("✅ Self-healing successful!")
                                    healing_log["fix_applied"] = True
                                    healing_log["new_implementation"] = {
                                        "code": new_code,
                                        "execution": fixed_execution
                                    }
                                    healing_log["attempts"].append({
                                        "attempt": attempt + 1,
                                        "analysis": debug_analysis,
                                        "votes": votes,
                                        "action": "code_regenerated",
                                        "consensus": "yes",
                                        "success": True
                                    })
                                    return healing_log
                                else:
                                    print(f"⚠️  Fix didn't work. Error: {fixed_execution.get('stderr', '')[:200]}")
                                    error_output = fixed_execution.get("stderr", "")
                                    healing_log["attempts"].append({
                                        "attempt": attempt + 1,
                                        "analysis": debug_analysis,
                                        "votes": votes,
                                        "action": "code_regenerated",
                                        "consensus": "yes",
                                        "success": False
                                    })
                        except Exception as e:
                            print(f"❌ Code regeneration failed: {e}")
                            healing_log["attempts"].append({
                                "attempt": attempt + 1,
                                "error": str(e)
                            })
                else:
                    print(f"⚠️  No consensus ({agreement_count}/3). Skipping fix.")
                    healing_log["attempts"].append({
                        "attempt": attempt + 1,
                        "analysis": debug_analysis,
                        "votes": votes,
                        "consensus": "no"
                    })
                    
            except Exception as e:
                print(f"❌ Healing attempt {attempt + 1} failed: {e}")
                healing_log["attempts"].append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        print("\n⚠️  Self-healing unsuccessful after all attempts.")
        return healing_log


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🧬 AI RESEARCH TEAM - LLM Training Efficiency Lab")
    print("="*80)
    
    team = AIResearchTeam()
    
    # Setup team
    team.setup_team()
    
    # Example research goal
    research_goal = """
Test if using a smaller learning rate warmup period (100 steps vs 1000 steps) 
affects training stability for a simple transformer model.

Hypothesis: Faster warmup may destabilize early training but reach similar final loss.
"""
    
    # Run sprint
    sprint_result = team.run_sprint(research_goal.strip())
    
    print("\n✅ Sprint completed! Check results/ directory for outputs.\n")


if __name__ == "__main__":
    main()
