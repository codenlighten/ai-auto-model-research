"""
Breakthrough Monitor
Watches for experiments rated ≥7/10 and triggers amplification
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

def monitor_for_breakthroughs(db_path="research_progress.db", threshold=7.0, check_interval=60):
    """Monitor database for breakthrough experiments"""
    
    print("🔍 Breakthrough Monitor Started")
    print(f"   Threshold: {threshold}/10")
    print(f"   Database: {db_path}")
    print(f"   Check interval: {check_interval}s\n")
    
    last_checked_id = None
    
    while True:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get latest experiment
            cursor.execute("""
                SELECT experiment_id, sprint_id, experiment_type, quality_rating, timestamp
                FROM experiments 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            
            if result and result[0] != last_checked_id:
                exp_id, sprint_id, exp_type, rating, timestamp = result
                last_checked_id = exp_id
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] New experiment: {sprint_id}")
                print(f"   Type: {exp_type}")
                print(f"   Rating: {rating if rating else 'N/A'}/10")
                
                if rating and rating >= threshold:
                    print(f"\n🎉 BREAKTHROUGH DETECTED! Rating: {rating}/10")
                    print(f"   Sprint: {sprint_id}")
                    print(f"   Type: {exp_type}")
                    print(f"\n🚀 TRIGGERING AMPLIFICATION TEAM...\n")
                    
                    # Trigger amplification
                    import subprocess
                    subprocess.run([
                        "python", 
                        "amplification_team.py",
                        "--sprint-id", sprint_id
                    ])
                    
                    print(f"\n✅ Amplification complete for {sprint_id}\n")
                else:
                    print(f"   Status: Continue researching (need {threshold}/10)\n")
            
            conn.close()
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(check_interval)


def get_current_stats(db_path="research_progress.db"):
    """Get current research statistics"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total = cursor.fetchone()[0]
        
        # Successful experiments
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'success'")
        successful = cursor.fetchone()[0]
        
        # Best rating
        cursor.execute("SELECT MAX(quality_rating) FROM experiments WHERE quality_rating IS NOT NULL")
        best_rating = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute("SELECT AVG(quality_rating) FROM experiments WHERE quality_rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0]
        
        # Breakthroughs (≥7)
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE quality_rating >= 7.0")
        breakthroughs = cursor.fetchone()[0]
        
        conn.close()
        
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("📊 RESEARCH LAB STATISTICS")
        print("="*60)
        print(f"Total Experiments:    {total}")
        print(f"Successful:           {successful} ({success_rate:.1f}%)")
        print(f"Best Rating:          {best_rating if best_rating else 'N/A'}/10")
        print(f"Average Rating:       {avg_rating:.1f}/10" if avg_rating else "Average Rating:       N/A")
        print(f"Breakthroughs (≥7):   {breakthroughs}")
        print("="*60)
        
        if breakthroughs > 0:
            print("\n🎉 Ready for amplification!")
        else:
            needed = 7.0 - (best_rating if best_rating else 0)
            print(f"\n⏳ Need {needed:.1f} more points to reach first breakthrough")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Error reading stats: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Run continuous monitor
        monitor_for_breakthroughs()
    else:
        # Show current stats
        get_current_stats()
