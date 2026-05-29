# EpiTADformer

EpiTADformer is a method for improved TAD boundary prediction utilizing positional information about epigenomic signals (transcription factor binding sites (TFBSs), histone modification marks). EpiTADformer is a transformer encoder-based model that learns both linear and non-linear positional relationships between epigenomic features associated with high-resolution (100 bp) genomic regions (bins).

![EpiTADtransformer main figure](method.png)

## System Requirement

### Software
- Python >= 3.10
- TensorFlow >= 2.15
- NumPy
- Pandas
- Scikit-learn

### Hardware
- 16 GB RAM or higher
- Multi-core CPU
- GPU with CUDA support (optional but recommended)


## Installation

### Environment setup 

Create a conda environment named `epitadformer`.

```bash
conda create -n epitadformer python=3.12 -y
conda activate epitadformer
```

### Package installation

- **Method1:** Install directly from the GitHub repository:

```bash
pip install git+https://github.com/dozmorovlab/EpiTADformer.git
```

- **Method2:** Install from a local clone and Install in editable mode:

```bash
git clonehttps://github.com/dozmorovlab/EpiTADformer.git
cd EpiTADformer
pip install -e .
```

- **Verify:** Verify installation

```bash
epitadformer-train --help
epitadformer-predict --help
epitadformer-data --help
```
If the installation is successful, each command will display its available options.

- *Update:* To install the latest version from GitHub:

```bash
pip install --upgrade git+https://github.com/dozmorovlabs/EpiTADformer.git
```


## Pipeline

### Step 1 – Data Preparation

EpiTADformer predicts TAD boundaries using epigenomic signal tracks. For each cell type of prediction interest, users must provide genomic signal data from relevant epigenomic features. Based on our feature importance analysis across multiple cell types, we identified the top 20 transcriptional and chromatin-associated features that contribute most strongly to TAD boundary prediction.

1. CTCF
2. ZZZ3
3. POLR2A
4. REST
5. YBX1
6. ZBTB11
7. RCOR1
8. EGR1
9. CEBPB
10. RAD21
11. TARDBP
12. ELF1
13. CREM
14. PML
15. EP300
16. UBTF
17. SMC3
18. TCF7
19. YY1
20. PKNOX1

> **Note:** Users do not need all 20 features. EpiTADformer can generate predictions with a subset of features (at least 5 features), although prediction performance may improve when more informative features are provided.

#### Input Requirements

Each epigenomic feature (for either training model in step 2 or prediction in step 3) should be provided as a separate tab-delimited bed/txt file containing **4 columns**:

| Column | Description |
|----------|-------------|
| chr | Chromosome name |
| start | Genomic start coordinate |
| end | Genomic end coordinate |
| value | Average Signal value for the genomic bin |

The default bin size used by EpiTADformer is **100 bp** resolution. All input features must be binned at the same resolution. For example,

```text
chr1    1000    1100    2
chr1    1100    1200    1
chr1    1200    1300    1
...
chr22   110000  110100  1
```

> **Note:** Place all feature files for a cell type into a single directory and order them alphabetically by feature name. Each file should contain genomic signal values for one epigenomic feature across the genome.


##### Process BAM signal input.
The signal files are commonly distributed as (unfiltered/filtered) allignment in BAM format. Signal files for many cell types can be download at ENCODEproject, GEO (Gene Expression Omnibus), 4D Nucleome Data Portal, etc. To process these files to the required format of EpiTAD former,
  - Generate binned signal tracks using bamCoverage from deepTools and export the results as a bedGraph file. For example,
  
```bash
  bamCoverage --bam <BAM file path> -o <output file path> --binSize 100 --outFileFormat bedgraph
```
  - If multiple BAM files are available for the same experiment, merge them before signal generation using samtools merge. For example,
  
```bash
samtools merge merged.bam replicate1.bam replicate2.bam replicate3.bam
```

> **Note:** The pretrained EpiTADformer models were developed using GM12878 epigenomic signals. We provided top 20 GM12878 features' proccessed files (100bp resolution) (by EpiTADformer required format), which can be download in Zenodo: https://doi.org/10.5281/zenodo.20442355. These includes a experiment per features, which was chosen as best among different labs' experimemts based on attention score analysis from our models.


### Step 2 - Train models.

#### Method 1.
We do provided pre-trained models (.h5) using Top 5, Top 7, Top 10, Top 15, top 20 of GM12878 epigenome features in 100 bp resolution which can be download in Zenodo: https://doi.org/10.5281/zenodo.20442355. Using this pre-trained model, user can go to **step 3** to make prediction on desired cell type or target's boundary. 

#### Method 2.
Beside the pre-trained models in **Method 1**, user can train model with different options of top GM12878 feature or top signal features in different cell lines.

Before training EpiTADformer models, the epigenomic signal files must be converted into deep-learning-ready datasets. This step is performed using the `epitadformer-data` command, which generates training, validation, and testing datasets stored as `.pkl` files.

##### Step 2 - 1: Prepare training, validation, and testing datasets 
Before training EpiTADformer models, the epigenomic signal files must be converted into deep-learning-ready datasets. This step is performed using the `epitadformer-data` command, which generates training, validation, and testing datasets stored as `.pkl` files.

Run:

```bash
epitadformer-data_train \
    --full_data_dir GM12878_signals \
    --truth_dir boundaries.bed \
    --peak_dir overlap_peaks \
    --save_dir training_data \
    --target_chrs chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22
```

  - Function parameters input:
  
  | Parameter | Description |
  |-------|------------------|
  | `--full_data_dir` | **Require** Directory containing all epigenomic feature files prepared in **Step 1**. |
  | `--truth_dir` | **Require** Directory of a BED file containing experimentally validated TAD boundary locations with 3 columns (chr, start, end).  |
  | `--peak_dir` | **Require** Overlap Peak Directory: Directory containing BED files folder of genomic regions that overlap strong epigenomic peaks (CTCF, RAD21, SMC3). This folder ('peakOverlap_chr') can be download in Zenodo: https://doi.org/10.5281/zenodo.20442355. |
  | `--save_dir` | **Require** Directory where generated `.pkl` datasets will be saved. |
  | `--target_chrs` | **Require** Chromosomes included in dataset generation. |
  | `--bin_size` | Genomic bin size used in the signal tracks. Default: `100`. |
  | `--sequence_len` | Number of genomic bins included in each input window. Must be an odd number. Default: `101`. |
  | `--num_iterations` | Number of independent datasets generated using different random seeds. Default: `20`. |
  
  ---
  

##### Step 2 - 2: Training iteration models 

After generating training datasets using **Step 2**, the next step is to train EpiTADformer models.

Run:

```bash
epitadformer-train \
    --data_dir training_data \
    --output_dir trained_models
```

  - Function parameters input:
  
  | Parameter | Description |
  |-------|------------------|
  | `--data_dir` | **Require** Directory containing `.pkl` training datasets generated by `epitadformer-data`. |
  | `--output_dir` | **Require** Directory where trained EpiTADformer models will be saved. |
  
  ---



### Step 3 - Prediction.

After obtain trained EpiTADformer models (**Step 2** - Method 1 or Method 2), users can predict TAD boundary bins and boundary regions in new cell types using their epigenomic signal tracks, which files needs to meet required format in **Step 1**.

Run:

```bash
epitadformer-predict \
  --full_data_dir GM12878_signals \
  --model_dir trained_models \
  --save_dir prediction_results \
  --target_chrs chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 \
  --min_files 10 \
  --bin_size 100 \
  --sequence_len 101
```

  - Function parameters input:

  | Parameter | Description |
  |-------|------------------|
  | `--full_data_dir` | **Require** Directory containing epigenomic signal files for prediction. Files must follow the EpiTADformer input format described in Step 1. |
  | `--model_dir` | **Require** Directory containing trained EpiTADformer `.h5` models. These may be user-trained models or pretrained models provided by EpiTADformer. |
  | `--save_dir` | **Require** Directory where prediction results will be written. |
  | `--target_chrs` | **Require** Chromosomes to analyze. Multiple chromosomes can be supplied. |
  | `--bin_size` | **Require** Genomic bin size used in the input feature files. Must match the resolution used during training. |
  | `--min_files` | Minimum number of models that must support a boundary prediction for it to be included in the final consensus result. Default = 16 |
  | `--sequence_len` | Number of genomic bins included in each prediction window. Must be an odd number. Default = 101|
  | `--save_iteration_models` | Save predictions generated by each individual model before consensus filtering. Default = False |

---


**Note:** Number and order of features used in trained models (window input dimension) will be same as those input used in prediction step 3

##### Prediction output layout

After prediction is completed, EpiTADformer generates the following output structure:

```text
prediction_results/
├── Consensus_BoundaryBins_ALL.bed
├── Consensus_Filtered_BoundaryRegion_All.bed
└── Iterations_models/                  (optional)
    ├── Binsbest_transformer_1_boundaryBin_ALL.bed
    ├── Binsbest_transformer_1_boundaryRegion_ALL.bed
    ├── Binsbest_transformer_2_boundaryBin_ALL.bed
    ├── Binsbest_transformer_2_boundaryRegion_ALL.bed
    └── ...
```

