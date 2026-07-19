#!/usr/bin/env bash
# ==============================================================================
# generate_next_dataset.sh
# 
# Helper script to run the OKF pipeline using the fine-tuned model and then
# generate the next iteration of the training dataset (v3).
# ==============================================================================

set -e

# Change directory to the script's directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "======================================================================"
echo "GENERATE NEXT TRAINING DATASET (v3)"
echo "======================================================================"
echo ""
echo "Select the execution mode:"
echo "  1) Fast Mode:   Run extraction on the syllabus seed only (17 chunks, ~1 min)"
echo "  2) Full Mode:   Run extraction on the full corpus (1006 chunks, ~1 hour)"
echo "  3) Skip Mode:   Skip extraction and use the current okf_results.json"
echo ""
read -rp "Enter choice [1, 2, or 3]: " choice

case "$choice" in
    1)
        echo ""
        echo "[1] Running pipeline on seed syllabus using the fine-tuned model..."
        python okf_pipeline.py pdfs/web_syllabi/AI_ML_Archipelago_Corpus_Seed.md
        ;;
    2)
        echo ""
        echo "[1] Running pipeline on the full corpus using the fine-tuned model..."
        echo "Note: This will take approximately 1 hour on an RTX 2050 GPU."
        python okf_pipeline.py
        ;;
    3)
        echo ""
        echo "[1] Skipping extraction stage..."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "[2] Running prepare_okf_training_data.py to clean and generate the v3 dataset..."
python prepare_okf_training_data.py

echo ""
echo "[3] Running split_okf_dataset.py to build v3 train/test splits..."
python split_okf_dataset.py

echo ""
echo "======================================================================"
echo "Dataset Generation Complete!"
echo "Files created in training_data/:"
echo "  - okf_training_pairs_v3.jsonl (Full cleaned records)"
echo "  - okf_train_pairs_v3.jsonl    (Train split)"
echo "  - okf_test_pairs_v3.jsonl     (Test split)"
echo "  - okf_dataset_report_v3.json  (Clean-up report)"
echo "  - okf_dataset_split_v3.json   (Splits metadata summary)"
echo "======================================================================"
