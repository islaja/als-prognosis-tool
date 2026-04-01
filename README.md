# ALS-Prognosis Docker Tool

This container provides a ready-to-use environment for running the
**ALS-prognosis-tool** inference pipeline. It simplifies the setup of complex
neuroimaging tools by bundling the MINC Toolkit, ANTs, Pelican, the SuStaIn and 
the Cox-net regularized regression models into a single, executable package.

This repository follows a Dockerfile-based distribution, ensuring that the computational environment, including all neuroimaging dependencies and pre-trained model weights, is perfectly reconstructed on your local machine, regardless of your host operating system.

## 📖 Scientific Basis

 Data include clinical and T1w images from CALSNIC, a large, prospectively acquired multicenter longitudinal ALS cohort ([Kalra et al., 2020](https://www.medrxiv.org/content/10.1101/2020.07.10.20142679v2)).

The preprocessing pipeline, the control dataset used for normalization,
and the trained SuStaIn model implemented in this tool are based on the
methodology described in [Lajoie et al., 2025a](https://doi.org/10.64898/2025.12.02.25341482).


> *Please refer to the article and its supplementary information for
> detailed documentation regarding the trained model parameters,
> harmonization techniques, and the control cohort used for w-scoring.*

This model was created using the [pySuStaIn framework](https://github.com/ucl-pond/pySuStaIn) based on the Subtype and Stage Inference algorithm ([Young et al., 2018](https://doi.org/10.1038/s41467-018-05892-0)). It was trained on regional deformation-based morphometry (DBM) w-scores from 14 anatomical regions selected to reflect the Brettschneider pTDP-43 staging scheme ([Brettschneider et al., 2013](https://doi.org/10.1002/ana.23937)) and established ALS imaging literature. These key regions include the motor and premotor cortices, the corticospinal tract and brainstem, fronto-parietal association cortices, basal ganglia, and medial temporal structures.

The survival prognosis is estimated using a Cox-net regularized regression. This model integrates the SuStaIn subtype/stage interaction with the individual's Disease Progression Rate (DPR) to project survival probability over time. The methodology to train the survival model and obtain the Individual Survival Distributions (ISDs) is described in [Lajoie et al., 2025b](https://doi.org/10.1002/ana.27196).

------------------------------------------------------------------------

## 🚀 Quick Start Guide

### 1. Clone this Repository
First, download the project files to your local machine:
```bash
git clone [https://github.com/islaja/als-prognosis-tool.git](https://github.com/islaja/als-prognosis-tool.git)
cd als-prognosis-tool
```

### 2. Install and Start Docker Desktop

https://www.docker.com/products/docker-desktop/

After installation, launch Docker Desktop and ensure it is running before continuing.

Recommended Docker resources:
- **CPUs:** 4 or more
- **Memory:** 8–10 GB

These can be configured in **Docker Desktop → Settings → Resources**.

------------------------------------------------------------------------
### 3. Build the Docker Image
Build the "engine" on your local machine. This process compiles the environment and only needs to be done once:
```bash
docker build --platform linux/amd64 -t als-prognosis-app .
```

------------------------------------------------------------------------
### 4. Prepare Your Input Data

#### CSV Manifest Format (`INPUT_FILE`)

Use the `Participant_Inputs_File_template.csv` located in the `examples/` folder as a template, and fill in with your own patients' information. 

Your CSV must contain these mandatory columns:

| Input | Definition |
| :--- | :--- |
| id | Unique participant identifier. |
| visit | Visit label (e.g., V1). |
| age | Participant age at time of scan. |
| sex | Biological sex. |
| scanner (optional but recommended) | Scanner model/ID used for harmonization. |
| dpr (optional) | Disease Progression Rate. If provided, it will be used directly for survival prediction. |
| symptom_duration_months (optional) | Symptom duration in months at the time of the ALSFRS assessment or MRI scan. Used to compute DPR if `dpr` is not provided. |
| alsfrs_total (optional) | Total ALSFRS score. Used together with `symptom_duration_months` to compute DPR if `dpr` is not provided: `(48 - alsfrs_total) / symptom_duration_months`. |
| path | File path to the subject's T1w or DBM map (`.nii`, `.nii.gz` or `.mnc`), relative to the `INPUT_DIR` folder. |

**Notes:** 

**1. Survival prediction and DPR**

Survival prediction requires a **Disease Progression Rate (DPR)**.

- If `dpr` is provided, it will be used directly.
- If `dpr` is not provided, it will be computed as:

  DPR = (48 − alsfrs_total) / symptom_duration_months

- If neither `dpr` nor the required variables (`alsfrs_total` and `symptom_duration_months`) are provided, **the survival prediction step will be skipped**.

**2. Scanner column (optional but recommended)**

Providing the `scanner` column improves the precision of the w-score computation by modeling and removing scanner-related effects during harmonization.

If provided, the value must match one of the supported scanner identifiers:

- `gedisc` — GE 3T Discovery MR 750  
- `philipsach` — Philips 3T Achieva TX  
- `philipsint` — Philips 3T Intera  
- `siemenspri` — Siemens 3T Prisma  
- `siemensprifit` — Siemens 3T Prisma Fit  
- `siemenstim` — Siemens 3T TIM Trio


#### Example Templates

Example CSV manifest files are provided in the examples/ folder of this
repository:

-   Participant_Inputs_File_t1w_example.csv: Example for raw
    T1-weighted inputs.
-   Participant_Inputs_File_dbm_example.csv: Example for dbm
    maps inputs.

------------------------------------------------------------------------

### 5. Configure the Wrapper Script
Open the `run_als_prognosis_docker.sh` file in a text editor. Update the USER SETTINGS section at the top to match your local paths and data:

⚠️ Windows Users: This script requires Unix (LF) line endings. While we've configured the repository to handle this automatically, please ensure your text editor (e.g., VS Code) is set to LF mode if you manually edit the script variables.

| Parameter   | Description |
| :---   | :--- | 
| `INPUT_DIR` | The folder on your computer containing your T1w (or DBM) maps and manifest CSV file. |
| `INPUT_FILE` | The name of your manifest CSV (e.g., subjects_manifest.csv). |
| `INPUT_TYPE` | Specifies the starting point of your data: `t1w_maps` or `dbm_maps`. |
| `OUTPUT_DIR` | Where you want the results saved. |
| `OUT_SUBDIR` | Sub-output directory name, defaults to a timestamped folder. |
| `PELICAN_DIR` | Path to the Pelican backend data folder. Note: The tool will download ~3GB of data here on the first run. |

------------------------------------------------------------------------

### 6. Execute the Pipeline
Run the script from your terminal to begin processing:

```bash
bash run_als_prognosis_docker.sh
```
⚠️ Windows Users: You must run this command inside a WSL2 (Ubuntu) terminal.

------------------------------------------------------------------------

## 🧠 The Processing Pipeline

The tool adapts its workflow based on your provided input_type. If you provide Deformation Based Morphometry maps (dbm_maps), the tool skips the first step and begins directly with regional extraction.

1.  **DBM Generation (Pelican):**
    * **Only for `t1w_maps`:** Utilizes the [Pelican Longitudinal Processing Pipeline](https://github.com/VANDAlab/Preprocessing_Pipeline) to generate high-quality Deformation Based Morphometry (DBM) maps.
    * **For `dbm_maps`:** This step is **skipped**; the tool uses your provided maps as the direct input for the next stage.
    * **Note:** The container automatically downloads the required Pelican repository (hosted on [Zenodo](https://zenodo.org/records/17168419)) during the first run.

2.  **Regional Extraction:** Once DBM maps are generated, the pipeline extracts
    regional averages from the anatomical volumes. Beyond the 14 regions required for SuStaIn, this step computes averages for:
    * All Gray Matter (GM) regions from the **CerebrA atlas** ([Manera et al., 2020](https://doi.org/10.1038/s41597-020-0557-9)).
    * GM, WM and ventricle volumes from the **Allen atlas** ([Hawrylycz et al., 2012](https://doi.org/10.1038/nature11405)).
    * White Matter (WM) tracts from the **JHU atlas** ([Wakana et al., 2007](https://doi.org/10.1016/j.neuroimage.2007.02.049)).
    * Combined GM and WM "hand-knob" regions using a custom anatomical mask.  
    
    The **ICBM CSF probability mask** is employed to exclude sulci from the average computation to ensure signal purity. All templates, atlas maps, and label descriptions are available under the `/resources` folder.

3.  **W-Scoring:** These regional values are converted into w-scores
    (z-scores adjusted for age, sex, and scanner) based on the normative
    control dataset described in [Lajoie et al., 2025a](https://doi.org/10.64898/2025.12.02.25341482).

4.  **SuStaIn Inferences:** The 14 selected regional w-scores
    are input into the pre-trained SuStaIn model to determine the 
    disease subtype and stage ([Lajoie et al., 2025a](https://doi.org/10.64898/2025.12.02.25341482)). 

5. **Survival Prediction:** If a Disease Progression Rate (DPR) is provided
    (or computed from symptom duration and ALSFRS-R scores), the tool utilizes a Cox-net regularized regression. Following the methodology established in [Lajoie et al., 2025b](https://doi.org/10.1002/ana.27196), the model integrates the DPR with the SuStaIn subtype × stage interaction to estimate Individual Survival Distributions (ISDs) and median survival times. 

    **Technical Note on Performance:** Internal cross-validation indicates that this optimized feature set (DPR + subtype × stage interaction) yields a higher C-index than the clinical + DBM feature combination reported in Lajoie et al., 2025b. These performance gains were verified using a nested cross-validation framework to prevent data leakage during the SuStaIn subtyping and stage inference process.
    
While these stages are running, the tool organizes the generated data into a structured output directory for analysis.

------------------------------------------------------------------------

## 📂 Output Structure & Interpretation

For each subject in your cohort, the pipeline generates a dedicated subfolder containing intermediate neuroimaging maps and final diagnostic plots. A complete sample of these outputs is provided in the [examples/outputs/](./examples/outputs/) folder for reference.

### Directory Organization
```text
[OUTPUT_DIR]/[OUT_SUBDIR]/
├── results_summary.csv                     # 🚩 Main cohort summary
├── pipeline.log                            # 📄 Log of the ALS-Prognosis pipeline
└── [Subject_ID]_[Visit]/                   # 📂 Individual subject folder
    ├── [Subject_ID]_[Visit]_dbm.nii.gz     # 🧠 DBM map (from Pelican)
    ├── roi_means_*.csv                     # 📄 Atlas-specific raw DBM averages
    ├── roi_means_all_atlas.csv             # 📄 All atlas raw DBM averages
    ├── roi_wscores_all_atlas.csv           # 📄 All atlas DBM w-scores averages
    ├── prognosis.png.                      # 📉 Individual predicted Survival Distribution (ISD) curve
    ├── predicted_survival_curve.csv        # 📄 Survival probability at each time point
    └── pelican_outputs/                    # 📂 Pelican log and intermediates (omitted in example)
```

### 📋 Detailed File Descriptions, Interpretation & Examples

* 🚩 **[`results_summary.csv`](./examples/outputs/results_summary.csv)**: **Main Cohort Summary.** Aggregates final inferences for all participants, including:
    * **Median Survival Time:** Estimated time (months) to 50% survival probability mark.
    * **SuStaIn Subtype & Stage:** Predicted disease trajectory (0 for normal-appearing, subtypes 1,2 or 3) and current progression point (Stages 0-14).
    * **Subtype Probabilities:** Specific confidence scores for each possible disease subtype.

    <div style="overflow-x: auto; white-space: nowrap; border: 1px solid #ddd; border-radius: 4px; padding: 0px; margin: 5px 0; font-size: 0.85em;">

    | **Example of results_summary.csv** | | | | | | | | | | | |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | ID | Visit | DPR | predicted_median_survival_time(months) | inferred_subtype | prob_inferred_subtype | inferred_stage | prob_inferred_stage | prob_s1 | prob_s2 | prob_s3 |
    | **sub1** | Visit_1 | 0.54 | 15.6 | 2.0 | 0.976 | 13.0 | 0.129 | 0.021 | 0.976 | 0.003 |
    | **sub2** | Visit_1 | 0.5 | 33.1 | 1.0 | 0.935 | 2.0 | 0.24 | 0.935 | 0.058 | 0.007 |
    | **sub3** | Visit_1 | 0.93 | 11.3 | 2.0 | 0.938 | 14.0 | 0.124 | 0.053 | 0.938 | 0.009 |
    </div>

* 📄 **[`pipeline.log`](./examples/outputs/pipeline.log)**: Full record of the execution steps from DBM extraction to survival prediction.
* 📂 **`[Subject_ID]_[Visit]/`**: Individual Subject Folder
    * 🧠 **`[Subject_ID]_[Visit]_dbm.nii.gz`** — Deformation-Based Morphometry (DBM) map obtained from Pelican.
    * 📄 [`roi_means_all_atlas.csv`](./examples/outputs/sub1_Visit_1/roi_means_all_atlas.csv) — Atlas-specific raw DBM averages.
    * 📄 [`roi_wscores_all_atlas.csv`](./examples/outputs/sub1_Visit_1/roi_wscores_all_atlas.csv) — DBM w-scores adjusted for age, sex, and scanner (if provided). 
        * *A w-score of 0 represents a "typical" control; negative values indicate atrophy (deviation from the norm); positive values indicate expansion relative to the norm.*
    * 📉 **`prognosis.png`**: Individual Survival Distribution (ISD) Curve. Visualizes predicted survival vs. reference cohorts.
    
        <b>Example of Prognosis Plot</b>
        ![Prognosis Plot Example](./examples/outputs/sub1_Visit_1/prognosis.png)
        
        **Plot Legend:**
        * **Black Line:** Individual predicted survival distribution (ISD) calculated using the Cox-net model.
        * **Colored Lines:** CALSNIC training population curves, colored according to their respective attributed SuStaIn subtype (S0: Pink, S1: Olive, S2: Teal, S3: Purple).
        * **Vertical Dashed Line:** Points to the predicted median survival (50% probability). In this example, the subject is identified as Subtype 2, Stage 13, with a predicted median survival of 15.6 months.

    * 📄 **[`predicted_survival_curve.csv`](./examples/outputs/sub1_Visit_1/predicted_survival_curve.csv)** — Survival probability data at each time point (0–60 months).
    * 📂 `pelican_outputs/` — Pelican logs and intermediate registration files (omitted in example).

------------------------------------------------------------------------

## 🔄 Version Control & Updates

To update your local environment after ALS-prognosis-tool is updated, following these steps:

**Back up the `run_als_prognosis_docker.sh`:**

Back up the `run_als_prognosis_docker.sh` script with your personal paths before running the next step to avoid overwriting it.

**Pull the latest code changes:**

```bash
git pull origin main
```

**Rebuild the Docker image:**

Docker will automatically detect changes in the code or dependencies and update the necessary layers.

```bash
docker build --platform linux/amd64 -t als-prognosis-app .
```

------------------------------------------------------------------------

## 🧪 Testing & Reliability

This project uses `pytest` to validate the core neuroimaging pipeline, prioritizing high-risk components over superficial coverage.

### Key Validations
* **ROI Integrity:** Integration tests verify atlas mapping on real DBM data to ensure zero `NaN` values.
* **Pipeline Orchestration:** Employs **monkeypatching** to validate complex logic (Pelican, w-scoring, SuStaIn) without heavy external dependencies.
* **Config Parsing:** Ensures model descriptors and resource YAMLs are correctly interpreted across processing steps.

### Status
* **Current:** Core preprocessing and subtyping orchestration tests are **passing**.
* **Roadmap:** Survival prognosis module coverage is planned; the modular architecture allows for easy extension using existing mock patterns.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
See the LICENSE file for full details.


## ⚠️ Medical Disclaimer

This tool is intended for research purposes only.

It has not been approved or cleared by regulatory authorities such as the
U.S. Food and Drug Administration (FDA) or Health Canada for clinical use.

The outputs and prognostic estimates generated by this tool must not be used
as the sole basis for clinical decision-making and should be interpreted only
within a research context by qualified professionals.


## 🧠 Intended Use & Limitations

This software is designed to support research in neuroimaging and disease
progression modeling, particularly in the context of ALS.

The models are trained on specific datasets and may not generalize to all populations.
Outputs are probabilistic and subject to uncertainty inherent to machine learning models.
The tool is not a medical device and is not intended for diagnosis, treatment, or patient management.

Users are responsible for ensuring appropriate use and for validating results
within their own research or clinical frameworks.