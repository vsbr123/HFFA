# HFFA

A Strip Steel Surface Defect Detection Method.

## Trained Model

## Training and Validation

Our MAFDet is implemented based on the [mmdetection3.3.0](#).

- **Training (using NEU-DET dataset as an example):**
  
  ```bash
  python train.py configs/HFFA/hffa_r50_hgs_rem_fpn_NEU-DET.py
  ```

- **Validation (using NEU-DET dataset as an example):**
  
  ```bash
  python test.py configs/HFFA/hffa_r50_hgs_rem_fpn_NEU-DET.py
  ```
