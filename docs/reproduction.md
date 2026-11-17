# Setup and reproduction

## Environment

The thesis describes BayesFlow 2.0. The four uploaded checkpoint archives record **Keras 3.10.0** in their metadata. A complete original environment lockfile and the exact BayesFlow/PyDDM package versions are not available in this repository.

[`requirements.txt`](../requirements.txt) lists the dependencies found in the published Python files and notebooks. It is a starting inventory, not a validated environment lockfile. In particular, BayesFlow APIs and Keras serialization must be compatible with the historical scripts and saved models.

Create an isolated environment using a Python version supported by your chosen BayesFlow and PyTorch versions. A PowerShell setup is:

```powershell
git clone https://github.com/ChristopherXMag/GDDM-SBI-SINGLETRIAL-EEG.git
cd GDDM-SBI-SINGLETRIAL-EEG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:KERAS_BACKEND = "torch"
```

For bash/zsh, use `source .venv/bin/activate` and `export KERAS_BACKEND=torch`. Set the backend before importing Keras or starting Jupyter. The torch backend is a setup option consistent with the scripts' PyTorch imports; checkpoint compatibility still needs verification in the selected environment.

For MATLAB preprocessing, install EEGLAB and the plugins required by the scripts' MFF import and `pop_clean_rawdata` calls. Configure the montage location and dataset paths in each file before execution.

## Training from simulation

From the repository root:

```text
python models/EEGGDDMfinal2.py
python models/EEGGDDM_Behavonly2.py
```

These commands train networks and run additional diagnostics. Review simulation counts, training settings, and output paths before launching them. The code uses stochastic simulation without a complete experiment-wide seed configuration, so a rerun is not expected to reproduce saved weights exactly.

The final scripts retain their historical `BasicWorkflow`, `DeepSet`, and `CouplingFlow` arguments. If your installed BayesFlow rejects an argument, resolve the version mismatch before interpreting results. Installing a package successfully does not validate the original workflow.

## Opening the original notebooks

Launch `jupyter lab` from the repository root and use the [notebook index](../notebooks/README.md) to choose an analysis. The notebooks are preserved with their original source cells and stored outputs.

Because the scripts now live in subdirectories, run this setup cell **before the original cells**. It works when the kernel starts in the repository root or a subdirectory:

```python
from pathlib import Path
import os
import sys

cwd = Path.cwd().resolve()
root = next(
    p for p in (cwd, *cwd.parents)
    if (p / "models" / "EEGGDDMfinal2.py").is_file()
    and (p / "analysis").is_dir()
)
os.chdir(root)
for folder in ("models", "analysis"):
    sys.path.insert(0, str(root / folder))
```

Next, replace local CSV paths in **all relevant cells**, including repeated `pd.read_csv` calls. The checkpoint directory now matches the `./checkpoints_unified_joint/...` paths used by most notebooks. Some cells use other checkpoint locations; inspect each selected cell.

## Items to resolve before interpreting a rerun

| Historical detail | What to check |
| --- | --- |
| Missing model modules | The posterior-predictive notebooks import `EEGGDDMGPT5_new` or `EEGGDDMGPT5_new2`, neither of which was published in this repository. Identify the intended simulator and checkpoint pairing before adapting those imports; similarly named final scripts are not proof of equivalence. |
| Missing group-analysis module | One cell in `precovery_and_multiple_subjects.ipynb` imports `analyze_multiple_subjects`, while only `analyze_multiple_subjects2.py` is published. A later cell uses the published helper. |
| Feature order and dimension | Compare the empirical column selection with the selected model's training schema. See [data format](data.md). |
| Checkpoint provenance | Four model archives are retained, but no per-run manifest links them conclusively to a script and prior. See the [inventory](../checkpoints_unified_joint/README.md). |
| Adapter state | Inspect how the saved model and each notebook apply standardization; verify that inputs use the preprocessing and scaling expected by the trained approximator. |
| Simulator fallback samples | The final likelihood functions contain broad exception handling and substitute samples when simulation fails. Monitor solver failures before using simulations for validation. |
| Different visualization simulator | `trial_dynamics_visualization.py` uses illustrative trajectory/RT generation and an N200 relationship that differ from the final likelihood. Treat those plots as illustrations rather than exact posterior-predictive simulations. |
| Multi-subject labels | The helper uses the filename segment before the first underscore as the subject key. Multiple task files with the same prefix can share a key; review identifiers before combining conditions. |

The built-in recovery functions also deserve review before reuse: they reduce a NaN mask only over `axis=1` of a three-dimensional summary tensor, rather than explicitly over both trial and feature axes. This documentation pass does not alter that analysis logic.

## What this cleanup verifies

The organization change checks preservation of every original scientific file, Python syntax, notebook JSON structure, and local documentation links. It does not rerun neural-network training, load the BayesFlow checkpoints, execute MATLAB, or reproduce thesis figures and statistics. Saved notebook outputs remain historical outputs.
