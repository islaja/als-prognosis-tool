# ALS-SuStaIn Docker Tool

This container provides a ready-to-use environment for running the
**ALS-prognosis-tool** inference pipeline. It simplifies the setup of complex
neuroimaging tools by bundling the MINC Toolkit, ANTs, Pelican, the SuStaIn and 
the CoxnetSurvival models into a single, executable package.

## 📖 Scientific Basis

The preprocessing pipeline, the control dataset used for normalization,
and the trained SuStaIn model implemented in this tool are based on the
methodology described in:

> **Reference:** https://doi.org/10.64898/2025.12.02.25341482\
> *Please refer to the article and its supplementary information for
> detailed documentation regarding the trained model parameters,
> harmonization techniques, and the control cohort used for w-scoring.*

This model was created using the **pySuStaIn** framework ([ucl-pond/pySuStaIn](https://github.com/ucl-pond/pySuStaIn)) based on the Subtype and Stage Inference algorithm (doi: [10.1038/s41467-018-05892-0](https://doi.org/10.1038/s41467-018-05892-0)). It was trained on **CALSNIC 1-2 data** ([https://doi.org/10.1212/WNL.92.15_supplement.P1.4-010](https://doi.org/10.1212/WNL.92.15_supplement.P1.4-010)) using regional W-scores from **14 anatomical regions** that integrate Brettschneider pTDP-43 stages with ALS imaging literature. These regions include the motor/premotor cortices, corticospinal tract and brainstem, fronto-parietal association cortices, basal ganglia, and medial temporal structures.

###### TODO ADD HERE ABOUT THE PREDICTION MODEL

------------------------------------------------------------------------

## 🚀 Quick Start Guide

### 1. Clone this Repository
First, download the project files and the script to your local machine:
```bash
git clone [https://github.com/your-username/als-sustain-app.git](https://github.com/your-username/als-sustain-app.git)
cd als-sustain-app
```

### 2. Build the Docker Image
Build the "engine" on your local machine. This process compiles the environment and only needs to be done once:
```bash
docker build -t als-prognosis-app .
```

### 3. Configure the Wrapper Script
Open the run_als_prognosis_docker.sh file in a text editor (like VS Code or Notepad). Update the USER SETTINGS section at the top to match your local paths:

`--INPUT_DIR` | The folder on your computer containing your images and CSV.
`--OUTPUT_DIR` | Where you want the results saved.
`--OUT_SUBDIR` | Sub output directory, defaults to a timestamped folder.
`--INPUT_FILE` | The name of your manifest CSV (e.g., subjects.csv).
`--INPUT_TYPE` | Specifies your data stage: `t1w_maps`, `dbm_maps`, `regional_dbm`, or `regional_dbm_wscore`. |
`--PELICAN_DIR` | Path to the 'Pelican' data folder on your computer. Note: The tool will download ~3GB of data here during the first run.

### 4. Execute the Pipeline
Run the script from your terminal to begin processing:

```bash
bash run_als_prognosis_docker.sh
```
⚠️ Windows Users: You must run this command inside a WSL2 (Ubuntu) terminal. Ensure your text editor is set to Unix (LF) line endings to avoid script errors.

------------------------------------------------------------------------

## 📂 Output Structure & Interpretation

Results are organized by execution timestamp within your designated `OUTPUT_DIR`. For each subject in your cohort, the pipeline generates a dedicated subfolder containing intermediate neuroimaging maps and final diagnostic plots.

### Directory Organization
```text
results_20260306/
├── results_summary.csv                # 🚩 Main Cohort Summary
└── [Subject_ID]_[Visit]/              # Individual Subject Folder
    ├── vbm/
    │   └── ..._dbm.mnc                # 🧠 Deformation-Based Morphometry map
    ├── roi_means_all_atlas.csv        # Raw regional volumes/means
    ├── roi_wscores_all_atlas.csv      # Harmonized W-scores
    ├── prognosis.png                  # 📊 Visual Survival Projection
    └── predicted_survival_curve.csv   # Survival probability over time

### 1. Global Summary (`results_summary.csv`)
This file aggregates the final inferences for all participants. It is the primary file for your statistical analysis and includes:

* **SuStaIn Subtype & Stage:** The predicted disease trajectory (0 for normal-appearing, 1,2 or 3) and progression point (ranging from **0** to **14**).
* **Subtype Probabilities:** The probability/certainty scores for each possible disease subtype.
* **Median Survival Time:** The estimated time (in months) to the survival endpoint, calculated by the **Coxnet model**.

### 2. Key Subject Outputs
* **W-scores:** Regional neuroimaging values adjusted for **age, sex, and scanner (if provided)**. 
    * A **W-score of 0** represents a "typical" control. 
    * **Lower negative values** indicate increasing degrees of atrophy or deviation from the norm.
    * **Higher positive values** indicate expansion relative to the norm.
* **Prognosis Plot (`.png`):** A visual representation of the predicted survival distribution for the individual, overlayed to the CALSNIC ALS patients as references, providing a clear clinical projection of the disease trajectory over time.

------------------------------------------------------------------------

## 🧠 The Processing Pipeline

For users providing **T1-weighted scans** (`t1w_maps`), the tool
executes a fully automated pipeline:

1.  **DBM Generation (Pelican):** Utilizes the Pelican Longitudinal 
    Processing Pipeline
    (https://github.com/VANDAlab/Preprocessing_Pipeline)
    to generate high-quality Deformation Based Morphometry (DBM) maps. 
    The container automatically download the required 
    Pelican repository, hosted on Zenodo:
    https://zenodo.org/records/17168419.
2.  **Regional Extraction:** Once DBM maps are generated, the pipeline extracts
    regional averages from the anatomical volumes. Beyond the 14 regions required for SuStaIn, this step computes averages for:
    * All Gray Matter (GM) regions from the **CerebrA atlas** [ref].
    * GM and ventricle volumes from the **Allen atlas** [ref].
    * White Matter (WM) tracts from the **JHU atlas** [ref].
    * Combined GM and WM "hand-knob" regions using a **custom anatomical mask**.  
    The **ICBM CSF probability mask** was employed to exclude sulci from the average computation to ensure signal purity. All templates, atlas maps, and label descriptions are available under the `/resources` folder.
3.  **W-Scoring:** These regional values are converted into W-scores
    (z-scores adjusted for age, sex, and scanner) based on the normative
    control dataset described in the referenced article.
4.  **SuStaIn Inferences:** The 14 selected regional W-scores*
    are input into the pre-trained SuStaIn model to determine the 
    disease subtype and stage. 
5.  **Survival Prediction:** If disease progression rate (DPR) (or both symptom duration and ALSFRS score) was provided, 
    the pre-trained CoxnetSurvival model is applied on the DPR and the subtype x stage interaction term to predict the individual survival distribution curve, as well as the median survival time. 

------------------------------------------------------------------------

## 📋 Prerequisites

-   **Docker Desktop**: Must be installed and running
    (https://www.docker.com/products/docker-desktop/).
    -   CPUs: 4 or more.
    -   Memory (RAM): 8GB-10GB recommended.
    Adjust these settings in Docker Desktop \> Settings \> Resources.
-   **Disk Space**: Ensure at least 10GB of free space
-   **Internet Connection**: Required only for the first run to download
    the Docker image and the pre-trained Pelican weights (\~3.3GB).

------------------------------------------------------------------------

## 📄 Data & CSV Formatting

The input CSV file (situated in INPUT_FILE)

### input CSV Structure

Your CSV must contain these mandatory columns:

-   id: Unique participant identifier.
-   visit: Visit label (e.g., V1).
-   age: Participant age at time of scan.
-   sex: Biological sex.
-   scanner (optional but recommanded): Scanner model/ID (used for harmonization).
-   dpr (optional): Disease progression rate if available.
-   symptom_duration_months (optional): Symptom duration in month until the ALSFRS test (or MRI scan) date. 
    Will serve to compute DPR if not already provided: (48 - ALSFRS) / symptom_duration_months
-   alsfrs_total (optional): ALSFRS score.
    Will serve to compute DPR if not already provided: (48 - ALSFRS) / symptom_duration_months
-   path: The file path relative to the INPUT_DIR folder. 
    The file is either a mnc or nii for INPUT_TYPE of 't1w_maps' or 'DBM_maps'.

#### Scanner Column (Optional but Recommended)

The **Scanner** column in the input CSV is **optional**, but providing it is **recommended** as it increases the precision of the W-score computation by explicitly modeling and removing scanner-related effects during harmonization.

If provided, the scanner value should be one of the following supported options:

- `gedisc` : GE 3T Discovery MR 750
- `philipsach` : Philips 3T Achieva TX
- `philipsint` : Philips 3T Intera
- `siemenspri` : Siemens 3T Prisma
- `siemensprifit` : Siemens 3T Prisma Fit
- `siemenstim` : Siemens 3T TIM Trio

### Input Type Details

| Input Type | Path column should point to... |
| :--- | :--- |
| `t1w_maps` | Raw `.mnc` or `.nii` T1-weighted scan files. |
| `dbm_maps` | Pre-existing DBM maps in `.mnc` or `.nii` format. |
| `regional_dbm` | `.csv` files containing extracted regional DBM averages. |
| `regional_dbm_wscore` | `.csv` files containing pre-calculated regional W-scores. |

------------------------------------------------------------------------

## 📁 Example Templates

Example CSV files are provided in the examples/ folder of this
repository:

-   Participant_Inputs_File_t1w_example.csv: Template for raw
    T1-weighted inputs.
-   Participant_Inputs_File_dbm_example.csv: Template for dbm
    maps inputs.
-   Participant_Inputs_File_regional_wscores_example.csv:
    Template for tabular W-score inputs. Note: The Path column here
    points to individual subject .csv files containing the regional
    values.

------------------------------------------------------------------------
