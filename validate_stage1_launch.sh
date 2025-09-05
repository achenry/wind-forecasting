#!/bin/bash

# ============================================================================
# Stage 1 TACTiS Tuning Pre-Launch Validation Script
# ============================================================================
# This script validates all requirements before launching Stage 1 tuning
# Run this before submitting: sbatch tune_model_storm_awaken_p60_stage1.sh
# ============================================================================

echo "============================================================"
echo "Stage 1 TACTiS Tuning Pre-Launch Validation"
echo "============================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track validation status
VALIDATION_PASSED=true

# ----------------------------------------
# 1. Check Database Credentials
# ----------------------------------------
echo "1. Checking Database Credentials..."

DB_LOGIN_FILE="/user/taed7566/Forecasting/Docs/db_login"

if [ -f "$DB_LOGIN_FILE" ]; then
    echo -e "${GREEN}✓ Database login file exists: $DB_LOGIN_FILE${NC}"
    
    # Check if we can parse it
    if grep -q "^DB=" "$DB_LOGIN_FILE" && grep -q "^FQDN=" "$DB_LOGIN_FILE" && grep -q "^USER=" "$DB_LOGIN_FILE"; then
        echo -e "${GREEN}  ✓ Login file has required fields${NC}"
        
        # Parse and display (without password)
        DB_HOST=$(grep "^FQDN=" "$DB_LOGIN_FILE" | cut -d'=' -f2)
        DB_USER=$(grep "^USER=" "$DB_LOGIN_FILE" | cut -d'=' -f2 | cut -d':' -f1)
        echo -e "  Database: ${DB_HOST}"
        echo -e "  User: ${DB_USER}"
    else
        echo -e "${RED}  ✗ Login file missing required fields${NC}"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${RED}✗ Database login file not found: $DB_LOGIN_FILE${NC}"
    echo "  This file should contain database credentials"
    VALIDATION_PASSED=false
fi

# ----------------------------------------
# 2. Check Data Files
# ----------------------------------------
echo ""
echo "2. Checking Data Files..."

DATA_PATH="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/DATA/preprocessed_awaken_data/awaken_processed_normalized.parquet"
NORM_PATH="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/DATA/preprocessed_awaken_data/awaken_processed_normalization_consts.csv"

if [ -f "$DATA_PATH" ]; then
    SIZE=$(ls -lh "$DATA_PATH" | awk '{print $5}')
    echo -e "${GREEN}✓ Data file exists: $DATA_PATH (${SIZE})${NC}"
else
    echo -e "${RED}✗ Data file not found: $DATA_PATH${NC}"
    VALIDATION_PASSED=false
fi

if [ -f "$NORM_PATH" ]; then
    echo -e "${GREEN}✓ Normalization constants file exists${NC}"
else
    echo -e "${RED}✗ Normalization file not found: $NORM_PATH${NC}"
    VALIDATION_PASSED=false
fi

# ----------------------------------------
# 3. Check Configuration Files
# ----------------------------------------
echo ""
echo "3. Checking Configuration Files..."

STAGE1_CONFIG="config/training/training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml"
CERT_PATH="config/certs/aiven_pg_ca.pem"

if [ -f "$STAGE1_CONFIG" ]; then
    echo -e "${GREEN}✓ Stage 1 config exists: $STAGE1_CONFIG${NC}"
    
    # Validate key settings (accounting for indentation in YAML)
    SKIP_COPULA=$(grep "^    skip_copula:" "$STAGE1_CONFIG" | awk '{print $2}')
    LOCK_SKIP=$(grep "^    lock_skip_copula:" "$STAGE1_CONFIG" | awk '{print $2}')
    MAX_EPOCHS=$(grep "^  max_epochs:" "$STAGE1_CONFIG" | awk '{print $2}')
    
    if [ "$SKIP_COPULA" = "true" ]; then
        echo -e "${GREEN}  ✓ skip_copula=true (correct for Stage 1)${NC}"
    else
        echo -e "${RED}  ✗ skip_copula should be true for Stage 1${NC}"
        VALIDATION_PASSED=false
    fi
    
    if [ "$LOCK_SKIP" = "true" ]; then
        echo -e "${GREEN}  ✓ lock_skip_copula=true (prevents automatic switching)${NC}"
    else
        echo -e "${RED}  ✗ lock_skip_copula should be true${NC}"
        VALIDATION_PASSED=false
    fi
    
    if [ "$MAX_EPOCHS" = "25" ]; then
        echo -e "${GREEN}  ✓ max_epochs=25 (Stage 1 marginals training)${NC}"
    else
        echo -e "${YELLOW}  ! max_epochs=$MAX_EPOCHS (expected 25 for Stage 1)${NC}"
    fi
else
    echo -e "${RED}✗ Stage 1 config not found: $STAGE1_CONFIG${NC}"
    VALIDATION_PASSED=false
fi

if [ -f "$CERT_PATH" ]; then
    echo -e "${GREEN}✓ PostgreSQL certificate exists${NC}"
else
    echo -e "${RED}✗ Certificate not found: $CERT_PATH${NC}"
    VALIDATION_PASSED=false
fi

# ----------------------------------------
# 4. Check SLURM Script
# ----------------------------------------
echo ""
echo "4. Checking SLURM Script..."

SLURM_SCRIPT="wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh"

if [ -f "$SLURM_SCRIPT" ]; then
    echo -e "${GREEN}✓ SLURM script exists: $SLURM_SCRIPT${NC}"
    
    # Check if executable
    if [ -x "$SLURM_SCRIPT" ]; then
        echo -e "${GREEN}  ✓ Script is executable${NC}"
    else
        echo -e "${YELLOW}  ! Script not executable, will make it executable${NC}"
        chmod +x "$SLURM_SCRIPT"
    fi
else
    echo -e "${RED}✗ SLURM script not found: $SLURM_SCRIPT${NC}"
    VALIDATION_PASSED=false
fi

# ----------------------------------------
# 5. Check Python Environment
# ----------------------------------------
echo ""
echo "5. Checking Python Environment..."

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    echo -e "${GREEN}✓ Mamba is available${NC}"
    
    # Check if environment exists
    if mamba env list | grep -q "wf_env_storm"; then
        echo -e "${GREEN}✓ wf_env_storm environment exists${NC}"
    else
        echo -e "${RED}✗ wf_env_storm environment not found${NC}"
        echo "  Create it with: mamba env create -f environment.yml"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${YELLOW}! Mamba not found in current shell${NC}"
    echo "  This is OK if running through SLURM which loads modules"
fi

# ----------------------------------------
# 6. Check Code Modifications
# ----------------------------------------
echo ""
echo "6. Checking Code Modifications..."

# Check for lock_skip_copula in TACTiS
if grep -q "lock_skip_copula" ../pytorch-transformer-ts/pytorch_transformer_ts/tactis_2/tactis.py 2>/dev/null; then
    echo -e "${GREEN}✓ lock_skip_copula parameter found in TACTiS model${NC}"
else
    echo -e "${RED}✗ lock_skip_copula not found in TACTiS model${NC}"
    echo "  The modifications may not be complete"
    VALIDATION_PASSED=false
fi

# Check for stage1_study in run_model.py
if grep -q "stage1_study" wind_forecasting/run_scripts/run_model.py 2>/dev/null; then
    echo -e "${GREEN}✓ stage1_study argument found in run_model.py${NC}"
else
    echo -e "${RED}✗ stage1_study not found in run_model.py${NC}"
    echo "  The modifications may not be complete"
    VALIDATION_PASSED=false
fi

# ----------------------------------------
# 7. Check Output Directories
# ----------------------------------------
echo ""
echo "7. Checking Output Directories..."

LOG_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs"
CHECKPOINT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/checkpoints"
OPTUNA_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/optuna"

for DIR in "$LOG_DIR" "$CHECKPOINT_DIR" "$OPTUNA_DIR"; do
    if [ -d "$DIR" ]; then
        echo -e "${GREEN}✓ Directory exists: $DIR${NC}"
    else
        echo -e "${YELLOW}! Creating directory: $DIR${NC}"
        mkdir -p "$DIR"
    fi
done

# ----------------------------------------
# 8. Test Database Connection (Optional)
# ----------------------------------------
echo ""
echo "8. Database Connection Test..."

if [ -f "$DB_LOGIN_FILE" ]; then
    echo "Testing PostgreSQL connection to Oldenburg University database..."
    
    # Parse credentials
    DB_HOST=$(grep "^FQDN=" "$DB_LOGIN_FILE" | cut -d'=' -f2)
    DB_PORT=$(grep "^DBPORT=" "$DB_LOGIN_FILE" | cut -d'=' -f2)
    DB_NAME=$(grep "^DB=" "$DB_LOGIN_FILE" | cut -d'=' -f2)
    USER_LINE=$(grep "^USER=" "$DB_LOGIN_FILE" | cut -d'=' -f2)
    DB_USER=$(echo "$USER_LINE" | cut -d':' -f1)
    DB_PASSWORD=$(echo "$USER_LINE" | cut -d':' -f2)
    
    # Try to connect using psql if available
    if command -v psql &> /dev/null; then
        export PGPASSWORD="$DB_PASSWORD"
        psql -h "$DB_HOST" \
             -p "$DB_PORT" \
             -U "$DB_USER" \
             -d "$DB_NAME" \
             -c "SELECT 1;" &>/dev/null
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Database connection successful${NC}"
        else
            echo -e "${YELLOW}! Could not connect to database${NC}"
            echo "  This might be OK if connection is only available from compute nodes"
        fi
    else
        echo -e "${YELLOW}! psql not available, skipping connection test${NC}"
        echo "  Connection will be tested when job runs"
    fi
else
    echo -e "${YELLOW}! Skipping database test (no credentials file)${NC}"
fi

# ----------------------------------------
# FINAL VALIDATION RESULT
# ----------------------------------------
echo ""
echo "============================================================"

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✓ ALL VALIDATIONS PASSED!${NC}"
    echo ""
    echo "Ready to launch Stage 1 tuning with:"
    echo ""
    echo "  cd /fs/dss/home/taed7566/Forecasting/wind-forecasting"
    echo "  sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh"
    echo ""
    echo "Configuration Summary:"
    echo "  - Model: TACTiS"
    echo "  - Dataset: AWAKEN (normalized)"
    echo "  - Context: 600s, Prediction: 60s"
    echo "  - Stage 1: Marginals only (25 epochs)"
    echo "  - GPUs: 3 parallel workers"
    echo "  - Trials per worker: 50"
    echo "  - Total trials: 100"
    echo ""
    echo "After completion, check logs for the study name to use in Stage 2."
else
    echo -e "${RED}✗ VALIDATION FAILED${NC}"
    echo ""
    echo "Please fix the issues above before launching."
    echo "Common fixes:"
    echo "  1. Check database credentials file: /user/taed7566/Forecasting/Docs/db_login"
    echo "  2. Ensure data files are preprocessed"
    echo "  3. Check that code modifications are in place"
fi

echo "============================================================"

# Return exit code based on validation
if [ "$VALIDATION_PASSED" = true ]; then
    exit 0
else
    exit 1
fi