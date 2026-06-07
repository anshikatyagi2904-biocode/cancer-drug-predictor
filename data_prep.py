import pandas as pd

# Load GDSC2 data
gdsc = pd.read_csv('/Users/ansh1kac0re/cancer-drug-predictor/data/gdsc2_raw.csv')

# See what we have
print("Shape:", gdsc.shape)
print("\nColumns:", gdsc.columns.tolist())
print("\nFirst 3 rows:")
print(gdsc.head(3))
print("\nUnique drugs:", gdsc['DRUG_NAME'].nunique())
print("Unique cell lines:", gdsc['CELL_LINE_NAME'].nunique())