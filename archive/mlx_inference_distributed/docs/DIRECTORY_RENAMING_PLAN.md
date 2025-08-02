# 📁 Directory Renaming & Reorganization Plan

## Rename for Clarity
Making directory names more descriptive:

1. `mlx` → `mlx_inference` (single-device inference)
2. `mlx_distributed` → `mlx_distributed_inference` (Team A)

---

## 🎯 Complete Reorganization Plan

### Step 1: Rename Directories
```bash
# Rename single-device inference
mv /Users/mini1/Movies/mlx \
   /Users/mini1/Movies/mlx_inference

# Rename Team A's distributed inference
mv /Users/mini1/Movies/mlx_distributed \
   /Users/mini1/Movies/mlx_distributed_inference
```

### Step 2: Move Team C's Projects Out
```bash
# Move RLHF project to top level
mv /Users/mini1/Movies/mlx_distributed_inference/mlx_rlhf \
   /Users/mini1/Movies/mlx_rlhf

# Remove empty directory and move KD project
rmdir /Users/mini1/Movies/mlx_knowledge_distillation
mv /Users/mini1/Movies/mlx_distributed_inference/mlx_knowledge_distillation \
   /Users/mini1/Movies/mlx_knowledge_distillation
```

### Step 3: Clean Up Deprecated Directory
```bash
# Remove Team B's old directory (already archived)
rm -rf /Users/mini1/Movies/mlx_training
```

---

## 📊 Final Structure

```
/Users/mini1/Movies/
├── mlx_inference/               # Single-device inference (port 8000)
├── mlx_distributed_inference/   # Team A: Distributed inference (port 8100)
├── mlx_distributed_training/    # Team B: Distributed training (port 8200)
├── mlx_rlhf/                   # Team C: RLHF implementation
└── mlx_knowledge_distillation/ # Team C: KD with RLHF integration
```

---

## ✅ Benefits of New Names

1. **Clear Purpose**: Each directory name describes what it does
2. **No Ambiguity**: "inference" vs "training" is explicit
3. **Team Ownership**: Clear which team owns what
4. **Logical Grouping**: Distributed projects are clearly marked

---

## 🔧 Post-Rename Updates Needed

### 1. Update Team A's imports/configs
```python
# In any Team A files that reference paths
# OLD: /Users/mini1/Movies/mlx_distributed/
# NEW: /Users/mini1/Movies/mlx_distributed_inference/
```

### 2. Update Single-Device Scripts
```bash
# In run_openai_server.py or similar
# OLD: cd /Users/mini1/Movies/mlx
# NEW: cd /Users/mini1/Movies/mlx_inference
```

### 3. Update Documentation
- DIRECTORY_STRUCTURE_GUIDE.md
- Any READMEs with path references
- Team coordination documents

---

## 🚀 Execution Order

1. First rename `mlx` → `mlx_inference`
2. Then rename `mlx_distributed` → `mlx_distributed_inference`
3. Move Team C's projects out
4. Clean up deprecated directories
5. Update all documentation

This gives us crystal clear directory names that match their actual purpose!