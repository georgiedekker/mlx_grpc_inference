# 🏗️ Team C Project Reorganization Plan

## Current Situation
Team C completed **TWO separate projects**, both currently nested inside Team A's directory:

1. **RLHF Implementation** → `/Users/mini1/Movies/mlx_distributed/mlx_rlhf/`
2. **Knowledge Distillation** → `/Users/mini1/Movies/mlx_distributed/mlx_knowledge_distillation/`

---

## 📁 Target Structure
Move both to top-level directories:

```
/Users/mini1/Movies/
├── mlx/                          # Original single-device
├── mlx_distributed/              # Team A (distributed inference)
├── mlx_distributed_training/     # Team B (distributed training)
├── mlx_rlhf/                    # Team C Project 1: RLHF ← MOVE HERE
└── mlx_knowledge_distillation/   # Team C Project 2: KD ← MOVE HERE
```

---

## 🚀 Reorganization Commands

### Step 1: Move RLHF Project
```bash
# Move Team C's RLHF project to top level
mv /Users/mini1/Movies/mlx_distributed/mlx_rlhf \
   /Users/mini1/Movies/mlx_rlhf
```

### Step 2: Move Knowledge Distillation Project
```bash
# First, remove the empty directory
rmdir /Users/mini1/Movies/mlx_knowledge_distillation

# Move Team C's KD project to top level
mv /Users/mini1/Movies/mlx_distributed/mlx_knowledge_distillation \
   /Users/mini1/Movies/mlx_knowledge_distillation
```

### Step 3: Clean Up Team A's Directory
```bash
# Remove Team C's installed packages from Team A's venv
rm -rf /Users/mini1/Movies/mlx_distributed/.venv/lib/python3.13/site-packages/mlx_knowledge_distillation*
rm -rf /Users/mini1/Movies/mlx_distributed/.venv/lib/python3.13/site-packages/mlx_rlhf*
```

### Step 4: Update Import Paths (if needed)
If any of Team A's code imports Team C's work:
```python
# OLD: from mlx_distributed.mlx_rlhf import ...
# NEW: from mlx_rlhf import ...
```

---

## ✅ Verification Steps

After reorganization, verify:

1. **Check new locations exist**:
   ```bash
   ls -la /Users/mini1/Movies/mlx_rlhf/
   ls -la /Users/mini1/Movies/mlx_knowledge_distillation/
   ```

2. **Verify Team C's projects still work**:
   ```bash
   # Test RLHF
   cd /Users/mini1/Movies/mlx_rlhf
   source .venv/bin/activate
   python -m pytest tests/

   # Test KD
   cd /Users/mini1/Movies/mlx_knowledge_distillation
   source ../.venv/bin/activate  # or create new venv
   python test_working_example.py
   ```

3. **Ensure Team A's directory is clean**:
   ```bash
   # Should not find any Team C files
   find /Users/mini1/Movies/mlx_distributed -name "*rlhf*" -o -name "*knowledge*"
   ```

---

## 📊 Final Result

Each team will have their own clean directories:

| Team | Project | Location | Status |
|------|---------|----------|--------|
| Original | Single-device inference | `/mlx/` | ✅ |
| Team A | Distributed inference | `/mlx_distributed/` | ✅ |
| Team B | Distributed training | `/mlx_distributed_training/` | ✅ |
| Team C | RLHF Implementation | `/mlx_rlhf/` | ✅ |
| Team C | Knowledge Distillation | `/mlx_knowledge_distillation/` | ✅ |

---

## ⚠️ Important Notes

1. **Team C has TWO separate projects** - both achieved high grades
2. **RLHF project** has its own venv and tests
3. **KD project** integrates with RLHF but is standalone
4. Both projects should maintain their independence
5. Update any documentation that references old paths

This reorganization will give Team C proper recognition for BOTH of their successful projects!