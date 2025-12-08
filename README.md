# ALS SuStaIn — Multi-model Inference

This repository provides a multi-model, modular pipeline to run SuStaIn inference on ALS patients.

## Quick start

1. Build the Singularity image (on a machine with Apptainer/Singularity installed):

```bash
apptainer build als_sustain.sif Singularity.def
```

2. Run the container on a CSV of T1s or feature CSVs (bind the data folder):

```bash
apptainer run --bind /data:/data als_sustain.sif run --model dbm_model --input /data/Participant_Inputs_File_t1.csv --workdir /data/out
```

3. Output will be printed as JSON. Results will also be stored in `--workdir` (customize pipeline to write files).

## Input file format

Two supported CSV formats (same columns: `ParticipantID,ParticipantVisit,Path`):

1. **Feature-based mode** — Path points to a single-row CSV with feature columns
2. **T1-based mode** — Path points to a T1w NIfTI (`.nii` or `.nii.gz`) that will be processed with PELICAN then ROI means

## Adding new models

Drop model files and a descriptor JSON in `als_sustain/models/` and add the model id to `als_sustain/config/models_catalog.json`. Descriptor JSON must include fields `model_id`, `model_file`, `input_type`, and `preprocessing`.
