# Census Income Classification & Segmentation

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn
```

## Running

Classification (predicts >50k vs <50k income):
```bash
python classification.py
```
Outputs: `classification_report.txt`, `classification_roc.png`, `classification_feature_importance.png`

Segmentation (groups population into marketing segments):
```bash
python segmentation.py
```
Outputs: `segmentation_report.txt`, `segmentation_svd.png`

## Files

- `classification.py` - random forest classifier
- `segmentation.py` - kmeans clustering
- `Project_Report.md` - writeup with findings and recommendations
- `census-bureau.data` - the dataset
- `census-bureau.columns` - column names
