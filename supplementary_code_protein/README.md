# Protein Supplementary Code Bundle

This folder contains the protein optimization code corresponding to the
supplementary DNA release. It includes the main optimization entry points and
the helper modules they directly import, but it does not include checkpoints,
training data, or result files.

Included:

- `optimize_final_real.py`: protein optimization pipeline used for TEM-style tasks
- `optimize_final_kras.py`: KRAS optimization pipeline
- `model/`: GRACE helper code
- `utils/`: amino-acid encoding helpers
- `oracles/`: small oracle wrapper modules used by the protein optimizers

This folder follows the same pattern as the DNA folder:

- code lives in the repository folder
- heavy assets such as checkpoints and results can be hosted separately

## Requirements

```bash
pip install -r requirements.txt
```
