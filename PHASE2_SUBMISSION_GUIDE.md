# PHASE-2 SUBMISSION GUIDE

## How to Run Phase-2 Inference

### Single Command:
```powershell
python src/hackathon_test_dataset_prediction.py
```

## Generated Files

After running, the following Phase-2 artifacts will be created in `results/`:

1. **phase2_prediction_log.txt**
   - Complete console output log
   - Full inference run details
   - All metrics and confusion matrix

2. **phase2_metrics.txt**
   - Structured metrics summary
   - Model information
   - Accuracy, Precision (macro), Recall (macro)
   - Correct predictions count

3. **phase2_confusion_matrix.txt**
   - Clean 9x9 confusion matrix table
   - Row = True labels
   - Column = Predicted labels

## Phase-2 Compliance Checklist

✅ Model: wafer_defect_model.onnx (UNCHANGED from Phase-1)
✅ Inference file: hackathon_test_dataset_prediction.py
✅ Engine: ONNX Runtime
✅ Post-processing: Confidence threshold = 0.3
✅ Metrics: Accuracy, Precision (macro), Recall (macro)
✅ Confusion matrix: 9x9 (including 'others')
✅ Console log: Captured to file
✅ Phase-1 files: UNTOUCHED

## Folder Structure (Phase-2 Only)

```
results/
├── phase2_prediction_log.txt       # NEW - Full console log
├── phase2_metrics.txt               # NEW - Structured metrics
└── phase2_confusion_matrix.txt     # NEW - Confusion matrix table
```

## Verification

Check that model is unchanged:
```powershell
# Model file should be exactly as submitted in Phase-1
dir models\wafer_defect_model.onnx
```

## Notes

- No retraining performed
- No model modification
- Inference-only post-processing
- All Phase-1 files remain intact
- Test dataset: 296 images (9 classes including 'others')

## Expected Output

- Test samples: 296
- Accuracy: ~30-40% (expected due to domain shift)
- Low-confidence predictions → 'others' class
- Confusion matrix shows all 9 classes

## Submission Package

Include these files in Phase-2 submission:
1. src/hackathon_test_dataset_prediction.py
2. results/phase2_prediction_log.txt
3. results/phase2_metrics.txt
4. results/phase2_confusion_matrix.txt
5. models/wafer_defect_model.onnx (unchanged from Phase-1)
