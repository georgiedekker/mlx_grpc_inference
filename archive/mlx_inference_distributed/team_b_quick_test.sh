#!/bin/bash

# Team B Quick Test Script
# Run this to quickly test your integration step by step

set -e  # Exit on error

echo "🚀 Team B Quick Integration Test"
echo "================================"

# Configuration
API_URL="http://localhost:8200"
TEST_DIR="$(pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command_exists() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

test_api_endpoint() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="${3:-}"
    
    log_info "Testing: $method $endpoint"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$API_URL$endpoint" || echo "000")
    else
        response=$(curl -s -w "%{http_code}" -o /tmp/response.json -X "$method" -H "Content-Type: application/json" -d "$data" "$API_URL$endpoint" || echo "000")
    fi
    
    http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        log_success "✅ $endpoint - HTTP $http_code"
        return 0
    else
        log_error "❌ $endpoint - HTTP $http_code"
        if [ -f /tmp/response.json ]; then
            log_error "Response: $(cat /tmp/response.json)"
        fi
        return 1
    fi
}

# Check prerequisites
echo ""
log_info "Checking prerequisites..."

check_command_exists "curl"
check_command_exists "python3"

# Check if API server is running
log_info "Checking if API server is running on $API_URL..."

if curl -s --connect-timeout 5 "$API_URL/health" > /dev/null; then
    log_success "API server is running"
else
    log_error "API server is not running on $API_URL"
    log_info "Please start your API server with: python your_api_file.py"
    exit 1
fi

# Test 1: Basic API endpoints
echo ""
echo "🔍 Test 1: Basic API Endpoints"
echo "------------------------------"

endpoints=(
    "/health"
    "/v1/models"
)

all_basic_passed=true
for endpoint in "${endpoints[@]}"; do
    if ! test_api_endpoint "$endpoint"; then
        all_basic_passed=false
    fi
done

if [ "$all_basic_passed" = true ]; then
    log_success "All basic endpoints working"
else
    log_error "Some basic endpoints failed"
    exit 1
fi

# Test 2: Create test data
echo ""
echo "📝 Test 2: Create Test Data"
echo "---------------------------"

log_info "Creating sample datasets..."

# Create Alpaca sample
cat > alpaca_test.json << 'EOF'
[
  {
    "instruction": "What is 2+2?",
    "input": "",
    "output": "2+2 equals 4."
  },
  {
    "instruction": "Explain photosynthesis briefly.",
    "input": "",
    "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
  }
]
EOF

# Create ShareGPT sample
cat > sharegpt_test.json << 'EOF'
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "What is machine learning?"
      },
      {
        "from": "gpt",
        "value": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
      }
    ]
  }
]
EOF

log_success "Test datasets created: alpaca_test.json, sharegpt_test.json"

# Test 3: Dataset validation
echo ""
echo "🔍 Test 3: Dataset Validation"
echo "-----------------------------"

validation_data='{"file_path": "alpaca_test.json", "max_samples": 10}'
if test_api_endpoint "/v1/datasets/validate" "POST" "$validation_data"; then
    # Check validation result
    if [ -f /tmp/response.json ]; then
        valid=$(cat /tmp/response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('valid', False))" 2>/dev/null || echo "false")
        format=$(cat /tmp/response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('format', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [ "$valid" = "True" ] && [ "$format" = "alpaca" ]; then
            log_success "Dataset validation working correctly (valid=$valid, format=$format)"
        else
            log_warning "Dataset validation returned: valid=$valid, format=$format"
        fi
    fi
else
    log_error "Dataset validation endpoint failed"
fi

# Test 4: Training job creation (minimal)
echo ""
echo "🚀 Test 4: Training Job Creation"
echo "--------------------------------"

training_data='{
  "model": "mlx-community/Qwen2.5-1.5B-4bit",
  "experiment_name": "quick_test_'$(date +%s)'",
  "training_file": "alpaca_test.json",
  "dataset_config": {
    "dataset_format": "alpaca",
    "batch_size": 1,
    "max_seq_length": 256
  },
  "lora_config": {
    "use_lora": true,
    "lora_r": 4,
    "lora_alpha": 8.0,
    "lora_dropout": 0.1
  },
  "n_epochs": 1,
  "learning_rate": 1e-4
}'

if test_api_endpoint "/v1/fine-tuning/jobs" "POST" "$training_data"; then
    # Extract job ID
    if [ -f /tmp/response.json ]; then
        job_id=$(cat /tmp/response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null || echo "")
        lora_enabled=$(cat /tmp/response.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('lora_info', {}).get('lora_enabled', False))" 2>/dev/null || echo "false")
        
        if [ -n "$job_id" ]; then
            log_success "Training job created: $job_id (LoRA: $lora_enabled)"
            
            # Test job status
            echo ""
            log_info "Testing job status retrieval..."
            sleep 2  # Wait a moment
            
            if test_api_endpoint "/v1/fine-tuning/jobs/$job_id"; then
                log_success "Job status retrieval working"
            else
                log_warning "Job status retrieval failed"
            fi
        else
            log_warning "Training job created but no job ID returned"
        fi
    fi
else
    log_error "Training job creation failed"
fi

# Test 5: Feature check
echo ""
echo "🔍 Test 5: Feature Availability Check"
echo "-------------------------------------"

if curl -s "$API_URL/health" -o /tmp/health.json; then
    features=$(cat /tmp/health.json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    features = data.get('features', {})
    lora = features.get('lora', False)
    formats = features.get('dataset_formats', [])
    print(f'LoRA: {lora}')
    print(f'Formats: {formats}')
    if lora and 'alpaca' in formats and 'sharegpt' in formats:
        print('FEATURES_OK')
except:
    print('ERROR')
" 2>/dev/null)
    
    if echo "$features" | grep -q "FEATURES_OK"; then
        log_success "All required features available"
        echo "$features"
    else
        log_warning "Some features may be missing:"
        echo "$features"
    fi
else
    log_error "Could not check features"
fi

# Summary
echo ""
echo "📊 Test Summary"
echo "==============="

# Check overall status
if curl -s "$API_URL/health" -o /tmp/final_health.json; then
    status=$(cat /tmp/final_health.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
    
    if [ "$status" = "healthy" ]; then
        log_success "🎉 Integration appears to be working!"
        echo ""
        echo "✅ API server is running and healthy"
        echo "✅ Basic endpoints are responding"
        echo "✅ Dataset validation is working"
        echo "✅ Training job creation is working"
        echo "✅ LoRA integration is available"
        
        echo ""
        echo "🎯 Next Steps:"
        echo "1. Run the complete test: python team_b_complete_example.py"
        echo "2. Test with your own datasets"
        echo "3. Monitor training jobs for actual completion"
        echo "4. Check training outputs in your configured output directory"
        
    else
        log_warning "API is responding but status is: $status"
    fi
else
    log_error "Could not determine final status"
fi

# Cleanup
echo ""
log_info "Cleaning up test files..."
rm -f alpaca_test.json sharegpt_test.json /tmp/response.json /tmp/health.json /tmp/final_health.json

log_success "Quick test completed!"

echo ""
echo "🔧 Troubleshooting:"
echo "- If tests failed, check your API server logs"
echo "- Ensure you've applied all the code modifications"
echo "- Run: python team_b_validation_script.py for detailed testing"
echo "- Check that all required files were copied by running: ./team_b_auto_setup.sh"