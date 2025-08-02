#!/bin/bash
# Team B Quick Test Script - Fast validation in 5 minutes

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Team B Quick Integration Test (5 min)              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Configuration
API_BASE="http://localhost:8200"
API_KEY="${MLX_TRAINING_API_KEY:-test-api-key}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Check API server
echo -e "\n${BLUE}1️⃣ Checking API server...${NC}"
if curl -s -f "${API_BASE}/docs" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API server is running${NC}"
else
    echo -e "${RED}❌ API server not running at ${API_BASE}${NC}"
    echo "Please start Team B's API server first"
    exit 1
fi

# Step 2: Test health endpoint
echo -e "\n${BLUE}2️⃣ Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s "${API_BASE}/health" 2>/dev/null || echo "{}")

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Health endpoint working${NC}"
    
    # Check for LoRA feature
    if echo "$HEALTH_RESPONSE" | grep -q '"lora".*true'; then
        echo -e "${GREEN}✅ LoRA feature reported${NC}"
    else
        echo -e "${YELLOW}⚠️  LoRA feature not reported in health check${NC}"
    fi
    
    # Check for dataset formats
    if echo "$HEALTH_RESPONSE" | grep -q "alpaca"; then
        echo -e "${GREEN}✅ Dataset formats reported${NC}"
    else
        echo -e "${YELLOW}⚠️  Dataset formats not reported in health check${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Health endpoint not implemented yet${NC}"
fi

# Step 3: Create minimal test dataset
echo -e "\n${BLUE}3️⃣ Creating test dataset...${NC}"
TEST_DATASET="/tmp/team_b_test_dataset.json"
cat > "$TEST_DATASET" << 'EOF'
[
    {
        "instruction": "What is LoRA?",
        "input": "",
        "output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the pre-trained model and injects trainable rank decomposition matrices."
    },
    {
        "instruction": "Why use LoRA?",
        "input": "",
        "output": "LoRA reduces memory usage by 90%, speeds up training, and produces small adapter files instead of full model copies."
    }
]
EOF
echo -e "${GREEN}✅ Test dataset created${NC}"

# Step 4: Test dataset validation
echo -e "\n${BLUE}4️⃣ Testing dataset validation...${NC}"
VALIDATE_RESPONSE=$(curl -s -X POST "${API_BASE}/v1/datasets/validate" \
    -H "Content-Type: application/json" \
    -d "{\"file_path\": \"$TEST_DATASET\"}" 2>/dev/null || echo "{}")

if echo "$VALIDATE_RESPONSE" | grep -q '"valid".*true'; then
    echo -e "${GREEN}✅ Dataset validation working${NC}"
    FORMAT=$(echo "$VALIDATE_RESPONSE" | grep -o '"format"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    echo "   Detected format: $FORMAT"
else
    echo -e "${YELLOW}⚠️  Dataset validation endpoint not implemented${NC}"
fi

# Step 5: Test training job creation
echo -e "\n${BLUE}5️⃣ Testing LoRA training job creation...${NC}"
JOB_RESPONSE=$(curl -s -X POST "${API_BASE}/v1/fine-tuning/jobs" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d "{
        \"experiment_name\": \"quick_test_$(date +%s)\",
        \"model_name\": \"mlx-community/Qwen2.5-1.5B-4bit\",
        \"epochs\": 1,
        \"lora\": {
            \"use_lora\": true,
            \"lora_r\": 8,
            \"lora_alpha\": 16.0
        },
        \"dataset\": {
            \"dataset_path\": \"$TEST_DATASET\",
            \"batch_size\": 2
        }
    }" 2>/dev/null || echo "{}")

if echo "$JOB_RESPONSE" | grep -q '"id"'; then
    JOB_ID=$(echo "$JOB_RESPONSE" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}✅ Training job created: $JOB_ID${NC}"
    
    if echo "$JOB_RESPONSE" | grep -q '"lora_enabled".*true'; then
        echo -e "${GREEN}✅ LoRA confirmed enabled${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Training endpoint needs LoRA integration${NC}"
fi

# Step 6: Summary
echo -e "\n${BLUE}📊 Quick Test Summary${NC}"
echo "══════════════════════════════════════════════════════════════"

TESTS_PASSED=0
TESTS_TOTAL=5

# Count passed tests
if curl -s -f "${API_BASE}/docs" > /dev/null 2>&1; then
    ((TESTS_PASSED++))
fi

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    ((TESTS_PASSED++))
fi

if echo "$VALIDATE_RESPONSE" | grep -q '"valid".*true'; then
    ((TESTS_PASSED++))
fi

if echo "$JOB_RESPONSE" | grep -q '"id"'; then
    ((TESTS_PASSED++))
fi

if echo "$JOB_RESPONSE" | grep -q '"lora_enabled".*true'; then
    ((TESTS_PASSED++))
fi

echo -e "Tests passed: ${TESTS_PASSED}/${TESTS_TOTAL}"

if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo -e "${GREEN}🎉 All tests passed! Team B integration is complete!${NC}"
elif [ $TESTS_PASSED -ge 3 ]; then
    echo -e "${YELLOW}⚠️  Most features working. Complete the remaining integrations.${NC}"
else
    echo -e "${RED}❌ Several features need implementation.${NC}"
fi

# Cleanup
rm -f "$TEST_DATASET"

echo -e "\n${BLUE}📝 Next Steps:${NC}"
if [ $TESTS_PASSED -lt $TESTS_TOTAL ]; then
    echo "1. Review team_b_api_modifications.py for missing endpoints"
    echo "2. Run python team_b_validation_script.py for detailed analysis"
    echo "3. Check team_b_training_logic.py for training integration"
else
    echo "1. Try training a real model with LoRA!"
    echo "2. Test with larger datasets"
    echo "3. Monitor memory usage reduction"
fi

echo -e "\n✅ Quick test completed in ~1 minute"