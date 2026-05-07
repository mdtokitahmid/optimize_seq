# RNA Supplementary Code Bundle

This folder contains the RNA optimization code corresponding to the
supplementary DNA release. It includes the mRNA optimization entry point and
the oracle helper code it depends on, but it does not include trained model
files, datasets, or results.

Included:

- `optimize_mrna.py`: mRNA optimization pipeline
- `model/`: GRACE helper code
- `Big_Oracles/mRNA/oracles/`: oracle helper code

This folder follows the same pattern as the DNA folder:

- code lives in the repository folder
- heavy assets such as trained model files and result files can be hosted separately

## Requirements

```bash
pip install -r requirements.txt
```
