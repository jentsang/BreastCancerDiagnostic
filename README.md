# Breast Cancer Diagnostics Dashboard

> An interactive diagnostic and visualization tool designed to compare morphological features between benign and malignant cases in breast cancer.

## Installation & Local Development

### 1. Clone the repository

```bash
git clone https://github.com/jentsang/BreastCancerDiagnostic.git
cd BreastCancerDiagnostic
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate breast_cancer_diagnostic
```

### 3. Run the dashboard

```bash
shiny run app.py --reload
```

### 4. Open in your browser

```
http://127.0.0.1:8000
```

### Dataset
The dashboard utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is publicly available through the following sources:

* **UCI Machine Learning Repository**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
* **Kaggle**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image.

---

### Contributors
* **Jennifer Tsang**
* **William Lee**
* **Abhinav Aggarwal**
* **Kevin Dong**
* **Lila Chan**

### License
This project is licensed under the MIT License.