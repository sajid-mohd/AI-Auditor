#!/bin/bash
set -e
echo "=== 1. Single audit (sanity check) ==="
python main.py audit data/sample_agents/customer_support.json results/audit_test.json

echo ""
echo "=== 2. Full eval suite ==="
python main.py eval

echo ""
echo "=== 3. Optimizer (baseline + 3 rounds) ==="
python main.py optimize

echo ""
echo "=== DONE ==="
cat results/comparison.md
