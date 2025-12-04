"""
Register the DebuggerAgent to the team
"""
from ai_research_team import AIResearchTeam

print("🔧 Registering DebuggerAgent...")

team = AIResearchTeam()
team.setup_team()

print("\n✅ Team setup complete with self-healing capabilities!")
print("\nAvailable agents:")
print("  1. ResearcherAgent - Research & hypothesis generation")
print("  2. ArchitectAgent - System design & architecture")
print("  3. CoderAgent - Python implementation")
print("  4. ValidatorAgent - Scientific validation")
print("  5. DebuggerAgent - Error analysis & self-healing")
