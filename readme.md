# HFFA

The Implementation of HFFA.


## Training and Validation

Our MAFDet is implemented based on the [mmdetection3.3.0](https://github.com/open-mmlab/mmdetection).

- **Training (using NEU-DET dataset as an example):**
  
  ```bash
  python train.py configs/HFFA/hffa_r50_hgs_rem_fpn_NEU-DET.py
  ```

- **Validation (using NEU-DET dataset as an example):**
  
  ```bash
  python test.py configs/HFFA/hffa_r50_hgs_rem_fpn_NEU-DET.py
  ```
