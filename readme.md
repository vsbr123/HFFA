# HFFA

A Strip Steel Surface Defect Detection Method.

## Trained Model

We provide pth of our HFFA trained  on the [neu-det](#) dataset: [neu-det.pth](#) (Code: xyty) ,on [gc10-det](#) dataset: [gc10-det.pth](#) (Code: a4aj), on the [cr7-det](#) dataset: [cr7-det.pth](#) (Code:........) and on the [spwd](#) dataset: [spwd.pth](#) (Code:........).

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
