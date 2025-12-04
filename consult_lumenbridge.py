"""
Consult Lumenbridge to help assemble AI computer scientist team
"""
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Lumenbridge API base URL
BASE_URL = "https://lumenbridge.xyz"

def consult_router(prompt):
    """
    Consult the ToolRouterAgent to get intelligent response
    """
    url = f"{BASE_URL}/api/router"
    
    payload = {
        "userPrompt": prompt
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\n{'='*80}")
    print("🌉 Consulting Lumenbridge ToolRouterAgent...")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✅ Success: {data.get('success')}")
        
        if 'routing' in data:
            print(f"\n📍 Routing Info:")
            print(f"   Selected Agent: {data['routing'].get('selectedAgent')}")
            print(f"   Reasoning: {data['routing'].get('reasoning')}")
            print(f"   Confidence: {data['routing'].get('confidence')}")
        
        print(f"\n📝 Result:")
        print(json.dumps(data.get('result'), indent=2))
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    # Our request to Lumenbridge
    prompt = """
I need help assembling a team of AI computer scientists, coders, and engineers to innovate on LLM training efficiency and model optimization.

PROJECT GOAL:
Build a collaborative team of AI agents that will:
1. Research novel approaches to training LLMs more efficiently
2. Design and test new model architectures
3. Implement optimizations in Python
4. Run real experiments in a sandbox environment
5. Validate results and iterate

REQUIREMENTS:
- Team should consist of specialized AI agents (researcher, architect, coder, validator, etc.)
- Agents need Python execution capabilities in a sandboxed folder
- They will work in 15-minute sprint cycles
- Must have ability to:
  * Create Python virtual environments
  * Install libraries using pip (we have SUDO_PASSWORD available)
  * Execute Python code and analyze results
  * Collaborate and discuss before implementing
  * Self-organize and plan their work

CONSTRAINTS:
- SearchAgent endpoint is currently broken, so avoid search-based approaches
- We prefer using Lumenbridge's existing agents (CodeGenerator, SchemaAgent, Schema Mediator, TerminalAgent)
- We can create custom user agents via /api/agents/register if needed
- System needs to be testable RIGHT NOW with real experiments

WHAT I NEED FROM YOU:
1. Recommend the optimal team structure (which specialist agents to create)
2. Suggest how to coordinate them for collaborative discussion before execution
3. Design the workflow for 15-minute sprint cycles
4. Identify what schemas we need for inter-agent communication
5. Recommend whether to use existing system agents or create custom user agents
6. Provide a concrete, actionable plan we can implement immediately

Please analyze this and give me your best recommendation for building this AI research team.
"""
    
    result = consult_router(prompt)
    
    if result:
        print(f"\n{'='*80}")
        print("💾 Saving full response to lumenbridge_team_recommendation.json")
        print(f"{'='*80}\n")
        
        with open('lumenbridge_team_recommendation.json', 'w') as f:
            json.dump(result, indent=2, fp=f)
        
        print("✅ Response saved!")
