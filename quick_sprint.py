"""
Quick Sprint Test - Run a single research sprint with existing agents
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_research_team import AIResearchTeam

def main():
    print("\n" + "="*80)
    print("🧪 QUICK SPRINT TEST")
    print("="*80)
    
    team = AIResearchTeam()
    
    # Simpler research goal that can run quickly
    research_goal = """
Test gradient clipping at different thresholds (0.5, 1.0, 5.0) 
on a simple 2-layer neural network.

Create a minimal experiment that trains for 10 steps and logs 
the gradient norms before and after clipping.
"""
    
    print("\n🎯 Research Goal:")
    print(research_goal.strip())
    print()
    
    # Check if agents exist
    try:
        agents = team.client.get_my_agents()
        print(f"✅ Found {agents.get('count', 0)} registered agents")
    except Exception as e:
        print(f"⚠️  Could not verify agents: {e}")
        print("Attempting to setup team...")
        team.setup_team()
    
    # Run the sprint
    sprint_result = team.run_sprint(research_goal.strip())
    
    # Show summary
    print("\n" + "="*80)
    print("📊 SPRINT SUMMARY")
    print("="*80)
    
    execution = sprint_result.get("phases", {}).get("implementation", {}).get("execution", {})
    if execution.get("success"):
        print("✅ Experiment executed successfully!")
        print(f"\n📈 Output (first 500 chars):")
        print(execution.get("stdout", "No output")[:500])
    else:
        print("❌ Experiment failed")
        print(f"Error: {execution.get('stderr', 'Unknown error')[:500]}")
    
    validation = sprint_result.get("phases", {}).get("validation", {})
    if "analysis" in validation:
        print(f"\n🔍 Validation Analysis (first 500 chars):")
        print(validation.get("analysis", "")[:500])
    
    print(f"\n📁 Full results: results/sprint_{sprint_result['sprint_id']}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
