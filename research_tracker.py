"""
Research Progress Tracker & Experiment Database
Tracks all experiments, builds knowledge base, identifies breakthroughs
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sqlite3

class ResearchDatabase:
    """SQLite database for tracking all experiments and progress"""
    
    def __init__(self, db_path="research_progress.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                sprint_id TEXT UNIQUE,
                experiment_type TEXT,
                research_goal TEXT,
                timestamp DATETIME,
                duration_seconds REAL,
                status TEXT,
                quality_rating INTEGER,
                breakthrough BOOLEAN DEFAULT 0
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                unit TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                insight_type TEXT,
                content TEXT,
                actionable BOOLEAN,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Best methods table (for publication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_methods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                method_name TEXT UNIQUE,
                experiment_id TEXT,
                efficiency_score REAL,
                description TEXT,
                code_path TEXT,
                publishable BOOLEAN DEFAULT 0,
                published_date DATETIME,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        self.conn.commit()
    
    def add_experiment(self, sprint_data):
        """Add experiment results to database"""
        cursor = self.conn.cursor()
        
        sprint_id = sprint_data.get('sprint_id')
        experiment_id = f"exp_{sprint_id}"
        
        # Extract basic info
        timestamp = sprint_data.get('start_time')
        duration = (datetime.fromisoformat(sprint_data.get('end_time')) - 
                   datetime.fromisoformat(timestamp)).total_seconds()
        
        # Determine experiment type from goal
        goal = sprint_data.get('goal', '')
        exp_type = self._classify_experiment(goal)
        
        # Get validation rating
        validation = sprint_data.get('phases', {}).get('validation', {})
        rating = self._extract_rating(validation.get('analysis', ''))
        
        # Check if successful
        implementation = sprint_data.get('phases', {}).get('implementation', {})
        status = 'success' if implementation.get('execution', {}).get('success') else 'failed'
        
        # Insert experiment
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (experiment_id, sprint_id, experiment_type, research_goal, timestamp, 
             duration_seconds, status, quality_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, sprint_id, exp_type, goal, timestamp, duration, status, rating))
        
        # Extract and store metrics
        self._extract_metrics(experiment_id, sprint_data)
        
        # Extract insights
        self._extract_insights(experiment_id, validation.get('analysis', ''))
        
        self.conn.commit()
        return experiment_id
    
    def _classify_experiment(self, goal):
        """Classify experiment type from goal"""
        goal_lower = goal.lower()
        if 'quantiz' in goal_lower:
            return 'quantization'
        elif 'prun' in goal_lower:
            return 'pruning'
        elif 'optim' in goal_lower:
            return 'optimizer'
        elif 'batch' in goal_lower:
            return 'batch_size'
        elif 'distill' in goal_lower:
            return 'distillation'
        elif 'precision' in goal_lower or 'amp' in goal_lower:
            return 'mixed_precision'
        elif 'checkpoint' in goal_lower:
            return 'gradient_checkpointing'
        elif 'architecture' in goal_lower:
            return 'architecture'
        else:
            return 'other'
    
    def _extract_rating(self, analysis):
        """Extract quality rating from validation analysis"""
        if not analysis:
            return None
        
        # Look for "Rating: X/10" pattern
        import re
        match = re.search(r'Rating:?\s*(\d+)/10', analysis, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_metrics(self, experiment_id, sprint_data):
        """Extract metrics from experiment output"""
        cursor = self.conn.cursor()
        
        execution = sprint_data.get('phases', {}).get('implementation', {}).get('execution', {})
        output = execution.get('stdout', '')
        
        # Parse common metrics from output
        import re
        
        # Look for final loss
        loss_matches = re.findall(r'(?:Final|final)?\s*[Ll]oss:?\s*(\d+\.?\d*)', output)
        if loss_matches:
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, unit)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'final_loss', float(loss_matches[-1]), 'loss'))
        
        # Look for time metrics
        time_matches = re.findall(r'(?:Time|time):?\s*(\d+\.?\d*)\s*(?:s|sec|seconds)', output)
        if time_matches:
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, unit)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'training_time', float(time_matches[0]), 'seconds'))
        
        # Look for speedup
        speedup_matches = re.findall(r'[Ss]peedup:?\s*(\d+\.?\d*)x', output)
        if speedup_matches:
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, unit)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'speedup', float(speedup_matches[0]), 'x'))
        
        # Look for compression ratio
        compression_matches = re.findall(r'[Cc]ompression:?\s*(\d+\.?\d*)x', output)
        if compression_matches:
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, unit)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'compression_ratio', float(compression_matches[0]), 'x'))
    
    def _extract_insights(self, experiment_id, analysis):
        """Extract key insights from validation analysis"""
        if not analysis:
            return
        
        cursor = self.conn.cursor()
        
        # Look for "Next Steps" section
        if "Next Steps:" in analysis or "next steps" in analysis.lower():
            cursor.execute("""
                INSERT INTO insights (experiment_id, insight_type, content, actionable)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'next_steps', analysis, True))
        
        # Look for limitations
        if "Limitations:" in analysis or "limitations" in analysis.lower():
            cursor.execute("""
                INSERT INTO insights (experiment_id, insight_type, content, actionable)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, 'limitations', analysis, True))
    
    def get_progress_summary(self):
        """Get overall research progress summary"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_experiments,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                AVG(CASE WHEN quality_rating IS NOT NULL THEN quality_rating END) as avg_rating,
                SUM(CASE WHEN quality_rating >= 7 THEN 1 ELSE 0 END) as publishable_quality,
                SUM(duration_seconds) / 3600.0 as total_hours
            FROM experiments
        """)
        
        row = cursor.fetchone()
        return {
            'total_experiments': row[0],
            'successful': row[1],
            'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
            'average_rating': row[2],
            'publishable_quality': row[3],
            'total_research_hours': row[4]
        }
    
    def get_best_methods(self, min_rating=7):
        """Get best performing methods for publication"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                e.experiment_id,
                e.experiment_type,
                e.quality_rating,
                e.research_goal,
                e.timestamp
            FROM experiments e
            WHERE e.quality_rating >= ? AND e.status = 'success'
            ORDER BY e.quality_rating DESC
        """, (min_rating,))
        
        return [dict(zip(['experiment_id', 'type', 'rating', 'goal', 'timestamp'], row)) 
                for row in cursor.fetchall()]
    
    def get_experiments_by_type(self, exp_type):
        """Get all experiments of a specific type"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT experiment_id, quality_rating, timestamp, status
            FROM experiments
            WHERE experiment_type = ?
            ORDER BY timestamp DESC
        """, (exp_type,))
        
        return cursor.fetchall()
    
    def mark_for_publication(self, experiment_id, method_name, efficiency_score):
        """Mark a method as ready for publication"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT research_goal FROM experiments WHERE experiment_id = ?
        """, (experiment_id,))
        
        goal = cursor.fetchone()[0]
        
        cursor.execute("""
            INSERT OR REPLACE INTO best_methods 
            (method_name, experiment_id, efficiency_score, description, publishable)
            VALUES (?, ?, ?, ?, 1)
        """, (method_name, experiment_id, efficiency_score, goal))
        
        self.conn.commit()
    
    def export_for_publication(self, output_dir="publications"):
        """Export publishable results to markdown and JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all publishable methods
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT bm.method_name, bm.efficiency_score, bm.description,
                   e.experiment_type, e.quality_rating, e.timestamp
            FROM best_methods bm
            JOIN experiments e ON bm.experiment_id = e.experiment_id
            WHERE bm.publishable = 1
            ORDER BY bm.efficiency_score DESC
        """)
        
        methods = cursor.fetchall()
        
        # Create publication document
        pub_file = output_path / f"research_findings_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(pub_file, 'w') as f:
            f.write("# Energy-Efficient AI Training Methods\n\n")
            f.write("## Research Findings from Premier AI Lab\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
            
            f.write("## Overview\n\n")
            summary = self.get_progress_summary()
            f.write(f"- Total Experiments: {summary['total_experiments']}\n")
            f.write(f"- Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"- Average Quality: {summary['average_rating']:.1f}/10\n")
            f.write(f"- Research Hours: {summary['total_research_hours']:.1f}h\n\n")
            
            f.write("## Breakthrough Methods\n\n")
            for method in methods:
                f.write(f"### {method[0]}\n\n")
                f.write(f"- **Type:** {method[3]}\n")
                f.write(f"- **Efficiency Score:** {method[1]:.2f}x\n")
                f.write(f"- **Quality Rating:** {method[4]}/10\n")
                f.write(f"- **Description:** {method[2][:200]}...\n\n")
        
        print(f"\n📄 Publication document created: {pub_file}")
        return pub_file
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def process_all_sprints(results_dir="results"):
    """Process all sprint results into database"""
    db = ResearchDatabase()
    results_path = Path(results_dir)
    
    sprint_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('sprint_')]
    
    print(f"\n📊 Processing {len(sprint_dirs)} sprints...")
    
    for sprint_dir in sprint_dirs:
        summary_file = sprint_dir / "sprint_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                sprint_data = json.load(f)
            
            exp_id = db.add_experiment(sprint_data)
            print(f"  ✅ {exp_id}: {sprint_data.get('phases', {}).get('validation', {}).get('analysis', '')[:50]}...")
    
    print("\n📈 Research Progress Summary:")
    summary = db.get_progress_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n🏆 Best Experiments (Rating ≥ 7):")
    best = db.get_best_methods()
    for exp in best:
        print(f"  • {exp['type']}: {exp['rating']}/10 - {exp['goal'][:60]}...")
    
    db.close()
    return summary


if __name__ == "__main__":
    print("🔬 Research Progress Tracker")
    print("=" * 80)
    
    # Process all existing sprints
    summary = process_all_sprints()
    
    # Create database connection
    db = ResearchDatabase()
    
    # Show progress
    print("\n" + "=" * 80)
    print("📚 Database ready for tracking future experiments")
    print("=" * 80)
    
    db.close()
