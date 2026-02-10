#!/bin/bash
#
# Sweep epsilon values for G2 identities analysis
# Tests epsilon values from 1e-10 to 1e-1 and saves results to separate directories
#

set -e  # Exit on error

# Number of points per run
N_POINTS=100

# Create base plots directory if it doesn't exist
mkdir -p plots

echo "=========================================="
echo "G2 Identities Epsilon Sweep"
echo "=========================================="
echo "Testing epsilon values: 1e-10 to 1e-1"
echo "Points per run: ${N_POINTS}"
echo "=========================================="
echo ""

# Loop through epsilon values from 1e-10 to 1e-1
for exp in {-10..-1}; do
    # Format exponent for directory name (remove minus sign)
    exp_abs=${exp#-}
    
    # Set epsilon value
    epsilon=$(python3 -c "print(10**${exp})")
    
    # Output directory
    output_dir="plots/eps-${exp_abs}"
    
    echo "----------------------------------------"
    echo "Running with epsilon = ${epsilon} (1e${exp})"
    echo "Output directory: ${output_dir}"
    echo "----------------------------------------"
    
    # Run the analysis
    python analysis/g2_identities_model_v2.py \
        --n-points ${N_POINTS} \
        --epsilon ${epsilon} \
        --output-dir ${output_dir}
    
    echo ""
    echo "Completed epsilon = ${epsilon}"
    echo ""
done

echo "=========================================="
echo "Epsilon sweep complete!"
echo "Results saved in:"
for exp in {-10..-1}; do
    exp_abs=${exp#-}
    echo "  plots/eps-${exp_abs}  (epsilon = 1e${exp})"
done
echo "=========================================="
