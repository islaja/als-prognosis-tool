# ALS-SuStaIn Docker Tool

This container provides a ready-to-use environment for running the
**ALS-SuStaIn** inference pipeline. It simplifies the setup of complex
neuroimaging tools by bundling the MINC Toolkit, ANTs, Pelican, and the SuStaIn
algorithm into a single, executable package.

## 📖 Scientific Basis

The preprocessing pipeline, the control dataset used for normalization,
and the trained SuStaIn model implemented in this tool are based on the
methodology described in:

> **Reference:** https://doi.org/10.64898/2025.12.02.25341482\
> *Please refer to the article and its supplementary information for
> detailed documentation regarding the trained model parameters,
> harmonization techniques, and the control cohort used for w-scoring.*

This model was created using the **pySuStaIn** framework ([ucl-pond/pySuStaIn](https://github.com/ucl-pond/pySuStaIn)) based on the Subtype and Stage Inference algorithm (doi: [10.1038/s41467-018-05892-0](https://doi.org/10.1038/s41467-018-05892-0)). It was trained on **CALSNIC 1-2 data** ([https://doi.org/10.1212/WNL.92.15_supplement.P1.4-010](https://doi.org/10.1212/WNL.92.15_supplement.P1.4-010)) using regional W-scores from **14 anatomical regions** that integrate Brettschneider pTDP-43 stages with ALS imaging literature. These regions include the motor/premotor cortices, corticospinal tract and brainstem, fronto-parietal association cortices, basal ganglia, and medial temporal structures.

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
4.  **SuStaIn Staging:** Finally, the 14 selected regional W-scores*
    are input into the pre-trained SuStaIn model to determine the 
    disease subtype and stage. 

------------------------------------------------------------------------

## 📋 Prerequisites

-   **Docker Desktop**: Must be installed and running
    (https://www.docker.com/products/docker-desktop/).
-   **Internet Connection**: Required only for the first run to download
    the Docker image and the pre-trained Pelican weights (\~3.3GB).

------------------------------------------------------------------------

## 🚀 How to Run

To process your data, use the following command in your terminal.
Replace `/path/to/your/data` with your actual local data folder.

``` bash
docker run --rm --platform linux/amd64 \
-v "/path/to/your/data:/app/data" \
-v "$HOME/.cache/als_sustain:/root/.cache/als_sustain" \
islaja/als-sustain-app:latest \
--model CALSNIC_sustain_14_reg_dbm_wscore \
--input_filepath /app/data/YOUR_INPUT_FILE.csv \
--input_type t1w_maps \
--outdir /app/data/output_results
```

## 🛠 Command Flags
| Flag | Description |
| :--- | :--- |
| `-v "/host/path:/app/data"` | **Volume Mount**: Connects your local data folder to the container's internal `/app/data` path. |
| `-v "...:/root/.cache/als_sustain"` | **Cache Mount**: Stores the 3.3GB Pelican models on your host machine to avoid re-downloading. |
| `--model` | **Currently only one accessible model:** `CALSNIC_sustain_14_reg_dbm_wscore`. This must be specified as shown. |
| `--input_filepath` | Internal path to your CSV file (e.g., `/app/data/subjects.csv`). |
| `--input_type` | Specifies your data stage: `t1w_maps`, `dbm_maps`, `regional_dbm`, or `regional_dbm_wscore`. |
| `--outdir` | Where to save the results (e.g., `/app/data/results`). |
| `--show_debug_outputs` | Set to `True` to see detailed logs and intermediate files. |

------------------------------------------------------------------------

## 📄 Data & CSV Formatting

The CSV file acts as a manifest for the pipeline. It must be placed
inside your local data folder.

### CSV Structure

Your CSV must contain these mandatory columns:

-   ID: Unique participant identifier.
-   Visit: Visit label (e.g., V1).
-   Age: Participant age at time of scan.
-   Sex: Biological sex.
-   Scanner: Scanner model/ID (used for harmonization).
-   Path: The file path relative to your mapped data folder.

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

## 🛠 Resource Management

Neuroimaging and deep learning (Pelican) are resource-intensive. For the
best experience, configure Docker to use:

-   CPUs: 4 or more.
-   Memory (RAM): 8GB minimum (16GB recommended).
-   Disk Space: Ensure at least 10GB of free space for the image and
    cached models.

Adjust these settings in Docker Desktop \> Settings \> Resources.
