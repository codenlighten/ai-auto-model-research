import sqlite3

conn = sqlite3.connect("research_progress.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT sprint_id, experiment_type, quality_rating, status, timestamp
    FROM experiments 
    ORDER BY quality_rating DESC 
    LIMIT 5
""")

print("\n🏆 TOP 5 EXPERIMENTS BY QUALITY RATING\n")
print("-" * 80)
for row in cursor.fetchall():
    sprint_id, exp_type, rating, status, timestamp = row
    print(f"{sprint_id:<25} | {exp_type:<30} | {rating if rating else 'N/A'}/10 | {status}")
print("-" * 80)

conn.close()
