#!/usr/bin/env bash
# Reversible Move-to-Trash script for approved Dharm candidate files
# Moves files to ~/.Trash/dharm_cleanup_<timestamp>/
set -euo pipefail

TRASH_DIR="$HOME/.Trash/dharm_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRASH_DIR"
echo "Moving approved retired weights to $TRASH_DIR..."

mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/bi_encoder"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1/bi_encoder/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/bi_encoder/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/bi_encoder"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-cpu-bs4/bi_encoder/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/bi_encoder/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/bi_encoder"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/bi_encoder/model.safetensors" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/bi_encoder/model.safetensors"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator/tokenizer.json" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1/cross_encoder_arbitrator/tokenizer.json"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator/tokenizer.json" "$TRASH_DIR/decision-corpus-ml/artifacts-cpu-bs4/cross_encoder_arbitrator/tokenizer.json"
mkdir -p "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator"
mv "/Users/siddharthgupta/Downloads/Documents/dharm/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator/tokenizer.json" "$TRASH_DIR/decision-corpus-ml/artifacts-v1.1-ablation-zeroshot/cross_encoder_arbitrator/tokenizer.json"

echo "Move completed successfully! Files can be restored from $TRASH_DIR if needed."
echo "Trash was NOT emptied automatically."
