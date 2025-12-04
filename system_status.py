"""
System Status Dashboard
Shows current state of the AI Research Organization
"""

import sqlite3
from datetime import datetime
import os

def show_dashboard():
    """Display comprehensive system status"""
    
    print("\n" + "="*80)
    print(" " * 20 + "🧠 AI RESEARCH ORGANIZATION STATUS")
    print("="*80)
    
    # === LAB CONFIGURATION ===
    print("\n📊 RESEARCH LAB (5 Agents)")
    print("-" * 80)
    agents = [
        ("ResearcherAgent", "Scientific analysis & hypothesis generation"),
        ("ArchitectAgent", "System design & architecture planning"),
        ("CoderAgent", "Implementation & code generation"),
        ("ValidatorAgent", "Quality control & validation"),
        ("DebuggerAgent", "Error analysis & self-healing")
    ]
    
    for name, role in agents:
        print(f"  ✓ {name:<20} {role}")
    
    print("\n  🔧 Self-Healing: ACTIVE (2/3 consensus voting)")
    print("  ⚡ Max Healing Attempts: 2 per experiment")
    
    # === AMPLIFICATION TEAM ===
    print("\n\n🚀 AMPLIFICATION TEAM (6 Agents)")
    print("-" * 80)
    amp_agents = [
        ("PhilosopherAgent", "Deep conceptual analysis & paradigm shifts"),
        ("StrategistAgent", "Strategic positioning & market timing"),
        ("ProductAgent", "Product vision & commercialization"),
        ("NarrativeAgent", "Storytelling & public communication"),
        ("DocumentalistAgent", "IP protection & documentation"),
        ("SynthesizerAgent", "Meta-analysis & insight synthesis")
    ]
    
    for name, role in amp_agents:
        status = "⏸️  Designed" if "Agent" in name else "✓"
        print(f"  {status} {name:<20} {role}")
    
    print("\n  🎯 Trigger: Experiments rated ≥ 7/10")
    print("  📝 Status: Designed, registration pending")
    
    # === RESEARCH STATISTICS ===
    try:
        conn = sqlite3.connect("research_progress.db")
        cursor = conn.cursor()
        
        # Total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total = cursor.fetchone()[0]
        
        # Successful
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'success'")
        successful = cursor.fetchone()[0]
        
        # Ratings
        cursor.execute("SELECT quality_rating FROM experiments WHERE quality_rating IS NOT NULL")
        ratings = [r[0] for r in cursor.fetchall()]
        
        # Breakthroughs
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE quality_rating >= 7.0")
        breakthroughs = cursor.fetchone()[0]
        
        # Latest experiment
        cursor.execute("""
            SELECT sprint_id, experiment_type, status, timestamp
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        latest = cursor.fetchone()
        
        conn.close()
        
        success_rate = (successful / total * 100) if total > 0 else 0
        best_rating = max(ratings) if ratings else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        print("\n\n📈 RESEARCH PERFORMANCE")
        print("-" * 80)
        print(f"  Total Experiments:     {total}")
        print(f"  Successful:            {successful} ({success_rate:.1f}%)")
        print(f"  Best Rating:           {best_rating}/10")
        print(f"  Average Rating:        {avg_rating:.1f}/10")
        print(f"  Breakthroughs (≥7):    {breakthroughs}")
        
        if latest:
            sprint_id, exp_type, status, timestamp = latest
            status_icon = "✅" if status == "success" else "⏳" if status == "running" else "❌"
            print(f"\n  Latest Experiment:")
            print(f"    {status_icon} {sprint_id}")
            print(f"    Type: {exp_type}")
            print(f"    Status: {status}")
        
        # Progress to breakthrough
        gap = 7.0 - best_rating
        if breakthroughs > 0:
            print(f"\n  🎉 BREAKTHROUGH ACHIEVED! Ready for amplification!")
        else:
            print(f"\n  ⏳ Need {gap:.1f} more quality points for first breakthrough")
            progress_bar = "█" * int((best_rating / 7.0) * 40)
            empty_bar = "░" * (40 - len(progress_bar))
            print(f"  Progress: [{progress_bar}{empty_bar}] {best_rating}/7.0")
        
    except Exception as e:
        print(f"\n  ⚠️  Could not load statistics: {e}")
    
    # === SYSTEM CAPABILITIES ===
    print("\n\n⚙️  SYSTEM CAPABILITIES")
    print("-" * 80)
    capabilities = [
        ("Autonomous Research", "Lab discovers novel ML techniques"),
        ("Self-Healing", "Debugger + consensus voting fixes errors"),
        ("ML Utilities", "Type-safe PyTorch helpers (SimpleCNN, prune_model, etc.)"),
        ("Breakthrough Amplification", "6-agent analysis creates impact packages"),
        ("Version Control", "Auto-commits results to GitHub"),
        ("Quality Assessment", "Validator rates each experiment /10")
    ]
    
    for capability, description in capabilities:
        print(f"  ✓ {capability:<25} {description}")
    
    # === NEXT STEPS ===
    print("\n\n🎯 NEXT STEPS")
    print("-" * 80)
    
    if breakthroughs > 0:
        print("  1. Run amplification team on breakthrough experiment")
        print("  2. Generate 6-perspective analysis package")
        print("  3. Synthesizer reveals meta-insights")
        print("  4. Publish impact package")
    else:
        print("  1. Complete current micro_architecture experiment")
        print("  2. Achieve first 7+/10 rating")
        print("  3. Trigger amplification system")
        print("  4. Demonstrate full end-to-end capability")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    show_dashboard()
