"""
Real-time Breakthrough Monitor
Watches database continuously for >= 7/10 experiments
"""

import sqlite3
import time
import os
from datetime import datetime

def monitor_breakthroughs(threshold=7.0, check_interval=30):
    """Monitor for breakthrough experiments in real-time"""
    
    print("\n" + "="*80)
    print("🔬 BREAKTHROUGH MONITOR ACTIVE")
    print("="*80)
    print(f"Threshold: {threshold}/10")
    print(f"Check interval: {check_interval}s")
    print(f"Database: research_progress.db")
    print("="*80 + "\n")
    
    last_check_time = None
    known_experiments = set()
    
    try:
        while True:
            conn = sqlite3.connect("research_progress.db")
            cursor = conn.cursor()
            
            # Get all experiments
            cursor.execute("""
                SELECT experiment_id, sprint_id, experiment_type, quality_rating, status, timestamp
                FROM experiments
                ORDER BY timestamp DESC
            """)
            
            experiments = cursor.fetchall()
            conn.close()
            
            # Check for new experiments
            current_time = datetime.now()
            new_breakthroughs = []
            new_experiments = []
            
            for exp in experiments:
                exp_id, sprint_id, exp_type, rating, status, timestamp = exp
                
                if exp_id not in known_experiments:
                    known_experiments.add(exp_id)
                    new_experiments.append((sprint_id, exp_type, rating, status))
                    
                    if rating and rating >= threshold:
                        new_breakthroughs.append((sprint_id, exp_type, rating))
            
            # Display new experiments
            if new_experiments:
                print(f"\n[{current_time.strftime('%H:%M:%S')}] NEW EXPERIMENTS:")
                for sprint_id, exp_type, rating, status in new_experiments:
                    rating_str = f"{rating}/10" if rating else "N/A"
                    status_icon = "✅" if status == "success" else "❌"
                    print(f"  {status_icon} {sprint_id} | {exp_type:<20} | {rating_str:<6} | {status}")
                
                # Alert on breakthroughs
                if new_breakthroughs:
                    print("\n" + "="*80)
                    print("🎉 " * 20)
                    print("BREAKTHROUGH DETECTED!")
                    print("🎉 " * 20)
                    print("="*80)
                    
                    for sprint_id, exp_type, rating in new_breakthroughs:
                        print(f"\n📊 Sprint: {sprint_id}")
                        print(f"🔬 Type: {exp_type}")
                        print(f"⭐ Rating: {rating}/10")
                    
                    print("\n" + "="*80)
                    print("🚀 READY TO TRIGGER AMPLIFICATION TEAM!")
                    print("="*80)
                    
                    # Show command to run
                    print(f"\nRun: python amplification_team.py --sprint-id {new_breakthroughs[0][0]}")
                    print()
            
            # Show stats periodically
            if last_check_time is None or (current_time - last_check_time).seconds >= 120:
                total = len(experiments)
                successful = sum(1 for e in experiments if e[4] == 'success')
                rated = [e[3] for e in experiments if e[3] is not None]
                best = max(rated) if rated else 0
                avg = sum(rated) / len(rated) if rated else 0
                breakthroughs = sum(1 for r in rated if r >= threshold)
                
                print(f"\n[{current_time.strftime('%H:%M:%S')}] STATS: "
                      f"{total} experiments | {successful} successful | "
                      f"Best: {best}/10 | Avg: {avg:.1f}/10 | "
                      f"Breakthroughs: {breakthroughs}")
                
                last_check_time = current_time
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user\n")
    except Exception as e:
        print(f"\n⚠️  Error: {e}\n")


if __name__ == "__main__":
    monitor_breakthroughs(threshold=7.0, check_interval=20)
