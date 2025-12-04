# Self-Healing System

## Overview

Your AI research team now has **autonomous error recovery** through the DebuggerAgent and consensus-based healing mechanism.

## How It Works

### 1. Error Detection

When an experiment fails during Phase 2 (Implementation):

- System captures error output (stderr + stdout)
- Preserves original code and context
- Triggers self-healing workflow

### 2. DebuggerAgent Analysis

The DebuggerAgent examines:

- **Stack trace**: Identifies exact error location
- **Error type**: ModuleNotFoundError, RuntimeError, TypeError, etc.
- **Context**: Original goal, code, and execution environment
- **Root cause**: Deep analysis of what went wrong

**Provides:**

1. Root Cause explanation
2. Fix Strategy (code or environment changes)
3. Terminal Commands (if needed)
4. Code Fix (corrected sections)
5. Confidence level (High/Medium/Low)

### 3. Team Consensus

Before applying any fix, the team votes:

- **ResearcherAgent**: Scientific validity of fix
- **ArchitectAgent**: System design implications
- **CoderAgent**: Implementation feasibility

**Consensus Rule:** 2 out of 3 must agree to proceed

### 4. Fix Application

If consensus reached:

**Environment Fixes:**

```powershell
# DebuggerAgent identifies missing package
pip install torch
# or
pip install pandas
```

**Code Fixes:**

- CoderAgent regenerates code with fix applied
- Incorporates DebuggerAgent's suggestions
- Uses ml_utils to prevent type errors
- Executes fixed code automatically

### 5. Validation

- If fix succeeds → Continue to Phase 3 (Validation)
- If fix fails → Try again (max 2 attempts)
- Logs all attempts for learning

## Example Workflow

```
Experiment Fails
     ↓
🔧 SELF-HEALING: Attempting error recovery...
     ↓
🔍 DebuggerAgent analyzes error
   "ModuleNotFoundError: No module named 'torch'"
   Root Cause: PyTorch not installed in virtual environment
   Fix: pip install torch
   Confidence: High
     ↓
📊 Team Consensus
   ResearcherAgent: AGREE - Standard dependency
   ArchitectAgent: AGREE - Required for ML experiments
   CoderAgent: AGREE - Missing package is clear issue
   ✅ Consensus reached (3/3 agree)
     ↓
📦 Installing packages: ['torch']
     ↓
🔨 Regenerating code with fix...
     ↓
🚀 Executing fixed code...
     ↓
✅ Self-healing successful!
```

## Common Errors Fixed

### 1. Missing Dependencies

**Error:**

```
ModuleNotFoundError: No module named 'torch'
```

**Fix:**

```powershell
pip install torch
```

### 2. PyTorch Type Errors

**Error:**

```
RuntimeError: expected scalar type Long but found Int
```

**Fix:**

```python
# Before
labels = torch.tensor(y)  # int32

# After
labels = torch.tensor(y).long()  # int64
```

### 3. Import Errors

**Error:**

```
ModuleNotFoundError: No module named 'ml_utils'
```

**Fix:**

- Copy ml_utils.py to experiment directory
- Already automated in system

### 4. Shape Mismatches

**Error:**

```
RuntimeError: size mismatch, m1: [32 x 784], m2: [128 x 10]
```

**Fix:**

```python
# Add missing layer or fix dimensions
x = x.view(x.size(0), -1)  # Flatten
```

## Configuration

**Max Healing Attempts:** 2

```python
healing_result = self._attempt_self_healing(
    implementation,
    discussion,
    sprint_dir,
    max_attempts=2  # Try twice
)
```

**Consensus Threshold:** 2/3 agents must agree

```python
if agreement_count >= 2:  # Majority
    print("✅ Consensus reached. Applying fix...")
```

## Benefits

### Before Self-Healing:

- 33% success rate
- Manual debugging required
- Experiments abandoned after first failure
- No learning from errors

### With Self-Healing:

- **Expected 60-80% success rate**
- Automatic error recovery
- Multiple fix attempts
- System learns common patterns
- Consensus prevents bad fixes

## Logging

All healing attempts are logged:

```json
{
	"self_healing": {
		"attempts": [
			{
				"attempt": 1,
				"analysis": "DebuggerAgent analysis...",
				"votes": {
					"ResearcherAgent": "AGREE - looks good",
					"ArchitectAgent": "AGREE - standard fix",
					"CoderAgent": "AGREE - will work"
				},
				"action": "code_regenerated",
				"consensus": "yes",
				"success": true
			}
		],
		"fix_applied": true
	}
}
```

## Future Enhancements

### Planned:

1. **Learning System**: Track common errors and preemptive fixes
2. **Auto-install**: Execute pip commands automatically
3. **Rollback**: Revert to previous working version
4. **Multi-strategy**: Try multiple fix approaches in parallel
5. **External Tools**: Use linters, type checkers for validation

### Advanced:

- GPU error recovery (CUDA OOM, device placement)
- Network error retry logic
- Data corruption detection
- Model checkpoint recovery

## Usage

Self-healing activates automatically when experiments fail:

```python
from ai_research_team import AIResearchTeam

team = AIResearchTeam()
team.setup_team()  # Registers DebuggerAgent

# Run experiment - healing happens automatically
result = team.run_sprint("Test batch size optimization")

# Check if healing was needed
if "self_healing" in result["phases"]:
    print("Self-healing was triggered!")
    print(f"Fix applied: {result['phases']['self_healing']['fix_applied']}")
```

## Team Members

1. **ResearcherAgent** - Scientific validity check
2. **ArchitectAgent** - System design review
3. **CoderAgent** - Implementation feasibility
4. **ValidatorAgent** - Results validation
5. **DebuggerAgent** - Error analysis & recovery ⭐ NEW

---

**Bottom Line:** Your AI team can now diagnose and fix its own errors through collaborative reasoning and consensus-based decision making. This significantly improves experiment success rates and reduces manual intervention! 🚀
