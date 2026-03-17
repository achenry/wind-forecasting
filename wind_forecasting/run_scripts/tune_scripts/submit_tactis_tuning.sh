#!/bin/bash
# Submit TACTiS-2 two-stage tuning with SLURM dependency chain
# Phase 2 auto-starts after Phase 1 completes successfully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

P1_JOB=$(sbatch --parsable "${SCRIPT_DIR}/tune_tactis_phase1_storm_smoothed_p60.sh")
echo "Phase 1: Job $P1_JOB submitted"

P2_JOB=$(sbatch --parsable --dependency=afterok:${P1_JOB} "${SCRIPT_DIR}/tune_tactis_phase2_storm_smoothed_p60.sh")
echo "Phase 2: Job $P2_JOB submitted (depends on Phase 1 job $P1_JOB)"
echo ""
echo "Monitor with: squeue -u \$USER"
