import pandas as pd

df = pd.read_csv("dataset_file_directory.csv", encoding="unicode_escape")

print("\n---- COLUMNS ----")
print(df.columns)

print("\n---- FIRST 20 ROWS ----")
print(df.head(20))

print("\n---- UNIQUE LABEL VALUES (first 50) ----")
print(df['Label'].unique()[:50])
