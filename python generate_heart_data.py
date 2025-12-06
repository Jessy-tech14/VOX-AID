import pandas as pd
import numpy as np

# Load audio label file
labels = pd.read_csv("dataset_file_directory.csv")

# Number of samples
n = len(labels)

np.random.seed(42)

# Generate synthetic heart rate data
synthetic = pd.DataFrame({
    "Filename": labels["Filename"],
    "Heart Rate BPM": np.random.normal(90, 10, n).astype(int),
    "Stress Level": np.random.choice(["Low", "Moderate", "High"], n),
    "Emotion Status": np.random.choice(["Calm", "Neutral", "Agitated"], n)
})

# Save file
synthetic.to_csv("synthetic_heart_rate_dataset.csv", index=False)
print("\n✅ Saved synthetic_heart_rate_dataset.csv with", n, "rows.")
