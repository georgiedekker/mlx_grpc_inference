# 🛡️ SAFE Directory Cleanup Plan - No Breaking!

## Current Mess:
- Team C's work is inside Team A's directory
- Unclear what Team B reimplemented vs what they lost
- Directory names don't clearly indicate purpose
- Old implementations mixed with new

---

## 🎯 Safe Cleanup Strategy: Test → Backup → Move → Verify

### Phase 1: Create Safety Backups (5 minutes)
```bash
# Create a safety backup of EVERYTHING first
cd /Users/mini1/Movies/
tar -czf MLX_BACKUP_$(date +%Y%m%d_%H%M%S).tar.gz mlx*
```

### Phase 2: Test Current Functionality (10 minutes)
Before moving anything, verify what's currently working:

```bash
# Test 1: Single-device inference (port 8000)
curl http://localhost:8000/v1/models 2>/dev/null || echo "Single-device not running"

# Test 2: Team A distributed inference (port 8100)  
curl http://localhost:8100/v1/models 2>/dev/null || echo "Team A not running"

# Test 3: Team B training API (port 8200)
curl http://localhost:8200/v1/fine-tuning/jobs 2>/dev/null || echo "Team B not running"
```

### Phase 3: Create Clean Structure - WITHOUT MOVING YET (5 minutes)
```bash
# Create the new structure empty first
mkdir -p /Users/mini1/Movies/MLX_CLEAN/{mlx_inference,mlx_distributed_inference,mlx_distributed_training,mlx_rlhf,mlx_knowledge_distillation}
```

### Phase 4: Copy (Don't Move!) One at a Time (15 minutes)

#### 4.1 Copy Single-Device Inference
```bash
cp -r /Users/mini1/Movies/mlx/* /Users/mini1/Movies/MLX_CLEAN/mlx_inference/
# Test it works from new location
cd /Users/mini1/Movies/MLX_CLEAN/mlx_inference
source .venv/bin/activate
python test_simple.py  # If this works, we're good!
```

#### 4.2 Copy Team A (without Team C's stuff)
```bash
# Copy Team A's files, excluding Team C's nested projects
rsync -av --exclude='mlx_rlhf' --exclude='mlx_knowledge_distillation' \
  /Users/mini1/Movies/mlx_distributed/ \
  /Users/mini1/Movies/MLX_CLEAN/mlx_distributed_inference/
```

#### 4.3 Copy Team B
```bash
cp -r /Users/mini1/Movies/mlx_distributed_training/* \
  /Users/mini1/Movies/MLX_CLEAN/mlx_distributed_training/
```

#### 4.4 Copy Team C's Projects
```bash
# RLHF project
cp -r /Users/mini1/Movies/mlx_distributed/mlx_rlhf/* \
  /Users/mini1/Movies/MLX_CLEAN/mlx_rlhf/

# Knowledge Distillation project  
cp -r /Users/mini1/Movies/mlx_distributed/mlx_knowledge_distillation/* \
  /Users/mini1/Movies/MLX_CLEAN/mlx_knowledge_distillation/
```

### Phase 5: Test Everything in New Location (10 minutes)
```bash
# Test each project works from its new location
# If ANY test fails, we still have the originals!

# Test 1: Single-device
cd /Users/mini1/Movies/MLX_CLEAN/mlx_inference
python test_simple.py

# Test 2: Team C's KD
cd /Users/mini1/Movies/MLX_CLEAN/mlx_knowledge_distillation
python test_working_example.py
```

### Phase 6: Switch Over (Only if ALL tests pass!)
```bash
# Rename old to backup
mv /Users/mini1/Movies/mlx /Users/mini1/Movies/mlx_OLD
mv /Users/mini1/Movies/mlx_distributed /Users/mini1/Movies/mlx_distributed_OLD
mv /Users/mini1/Movies/mlx_distributed_training /Users/mini1/Movies/mlx_distributed_training_OLD

# Move clean versions to final locations
mv /Users/mini1/Movies/MLX_CLEAN/* /Users/mini1/Movies/
rmdir /Users/mini1/Movies/MLX_CLEAN

# Final cleanup only after confirming everything works
# rm -rf /Users/mini1/Movies/mlx_OLD
# rm -rf /Users/mini1/Movies/mlx_distributed_OLD  
# rm -rf /Users/mini1/Movies/mlx_distributed_training_OLD
# rm -rf /Users/mini1/Movies/mlx_training  # deprecated
```

---

## 🚦 Stop Points - When NOT to Proceed

STOP if:
- ❌ Any test fails in Phase 2
- ❌ Copy operations show errors in Phase 4
- ❌ Tests fail in Phase 5
- ❌ You're unsure about any step

---

## 📊 Final Clean Structure

```
/Users/mini1/Movies/
├── mlx_inference/               # Single-device (was mlx)
├── mlx_distributed_inference/   # Team A only (was mlx_distributed)
├── mlx_distributed_training/    # Team B only
├── mlx_rlhf/                   # Team C project 1
├── mlx_knowledge_distillation/ # Team C project 2
└── [mlx_training - removed]    # Deprecated
```

---

## 🎯 Why This is Safe

1. **Full backup** before touching anything
2. **Copy, don't move** until verified
3. **Test at each step**
4. **Keep old versions** until 100% sure
5. **Clear rollback path** at every stage

---

## 💡 Quick Rollback if Needed

If ANYTHING goes wrong:
```bash
# Restore from backup
cd /Users/mini1/Movies/
tar -xzf MLX_BACKUP_[timestamp].tar.gz
```

This approach means you can clean up the mess with ZERO risk of breaking anything!