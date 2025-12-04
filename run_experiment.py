"""
Run Premier Research Lab with proper encoding settings
Prevents Unicode encoding errors in Windows
"""

import subprocess
import sys
import os

# Set UTF-8 encoding for subprocess
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUTF8'] = '1'

# Run the research lab with proper encoding
if len(sys.argv) < 2:
    print("Usage: python run_experiment.py <experiment_name>")
    sys.exit(1)

experiment_name = sys.argv[1]

print(f"\n🚀 Running experiment: {experiment_name}")
print(f"   Encoding: UTF-8")
print(f"   Python: {sys.executable}\n")

# Run premier_research_lab.py with the experiment
cmd = [sys.executable, "premier_research_lab.py", experiment_name]

process = subprocess.run(
    cmd,
    env=env,
    capture_output=False,  # Stream output directly
    text=True
)

sys.exit(process.returncode)
