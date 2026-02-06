#!/bin/bash

# Run all workflow steps for G2 Structure Learning
# This script executes the entire pipeline described in run_interactive.ipynb

set -e  # Exit on error

echo "====================================="
echo "Starting G2 Structure Learning Pipeline"
echo "====================================="

# Step 1: Train CY Metric Model
echo ""
echo "Step 1: Training CY Metric Model..."
python run_cy.py

# Step 2: Generate G2 Sample Data
echo ""
echo "Step 2: Generating G2 Sample Data..."
python sampling.py

# Step 3: Train G2 3-form Model
echo ""
echo "Step 3: Training G2 3-form Model..."
python run_g2.py --task 3form

# Step 4: Train G2 Metric Model
echo ""
echo "Step 4: Training G2 Metric Model..."
python run_g2.py --task metric

# Step 5: Run data analysis
echo ""
echo "Step 5: Running Data Analysis..."
python analysis/data_analysis.py

# Step 6: Validate CY Metric Kählerity
echo ""
echo "Step 6: Validating CY Metric Kählerity..."
python analysis/cy_kahlerity.py

# Step 7: Validate G2 Identities (Analytic)
echo ""
echo "Step 7: Validating G2 Identities (Analytic)..."
python analysis/g2_identities_analytic.py

# Step 8: Validate G2 Identities (Model)
echo ""
echo "Step 8: Validating G2 Identities (Model)..."
python analysis/g2_identities_model.py

echo ""
echo "====================================="
echo "Pipeline Complete!"
echo "====================================="
