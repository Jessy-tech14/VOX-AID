#!/usr/bin/env python3
"""
Multimodal training script (Audio + Heart-rate) with:
- audio feature extraction (MFCC + spectral features)
- robust dataset alignment (filename join or index-alignment fallback)
- sklearn Pipeline (numeric imputation + one-hot for categories)
- RandomForest training/evaluation (accuracy, report, confusion matrix)
- pop-up visualizations (waveform/spectrogram/MFCC + distributions + feature importance)
"""

import os
import glob
import math

import numpy as np
import pandas as pd
import librosa
import librosa.display

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import joblib

# --------------------------------------------------
# CONFIGURATION — update these paths only
# --------------------------------------------------
AUDIO_DIR = r"C:/Users/user/Desktop/voice/"
AUDIO_LABEL_CSV = r"C:/Users/user/Desktop/voice/dataset_file_directory.csv"
HEART_CSV = r"C:/Users/user/Desktop/voice/synthetic_heart_rate_dataset.csv"

MODEL_OUT = "multimodal_rf_model.joblib"
FEATURES_OUT = "multimodal_features.csv"

# Which target do you want to predict?
# Options (based on your merged columns): "label" (vocalization meaning), "Stress Level", "Emotion Status"
TARGET_COL = "label"

# Alignment mode: "auto" (try filename join, else index alignment)
ALIGN_MODE = "auto"
# --------------------------------------------------


# --------------------------------------------------
# VISUALIZATION HELPERS (POP-UP)
# --------------------------------------------------
def plot_waveform(path):
    y, sr = librosa.load(path, sr=22050)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {os.path.basename(path)}")
    plt.tight_layout()
    plt.show()

def plot_melspectrogram(path):
    y, sr = librosa.load(path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logS = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(logS, x_axis="time", y_axis="mel", sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {os.path.basename(path)}")
    plt.tight_layout()
    plt.show()

def plot_mfcc(path):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title(f"MFCC: {os.path.basename(path)}")
    plt.tight_layout()
    plt.show()

def plot_heart_rate_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Heart Rate BPM"], kde=True)
    plt.title("Heart Rate Distribution")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.show()

def plot_label_distribution(df, col):
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Distribution of target: {col}")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx], importances[idx])
    plt.title("Feature Importance (RandomForest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
def load_audio_labels(csv_path):
    df = pd.read_csv(csv_path)
    # Keep the key columns you already use (and allow extra columns to exist)
    if not {"Filename", "Label"}.issubset(df.columns):
        raise ValueError("dataset_file_directory.csv must contain at least columns: Filename and Label")
    return df[["Filename", "Participant", "Label"]].copy()

def load_heart_data(csv_path):
    df = pd.read_csv(csv_path)
    # We expect at least: Filename and Heart Rate BPM (others may exist)
    if "Filename" not in df.columns:
        raise ValueError("heart-rate CSV must contain a 'Filename' column.")
    if "Heart Rate BPM" not in df.columns:
        raise ValueError("heart-rate CSV must contain a 'Heart Rate BPM' column.")
    return df.copy()


# --------------------------------------------------
# AUDIO FEATURE EXTRACTION (robust to short clips)
# --------------------------------------------------
def _safe_n_fft(signal_length, max_n_fft=2048):
    # choose the largest power-of-two <= min(max_n_fft, signal_length)
    if signal_length <= 1:
        return 2
    max_allowed = min(max_n_fft, signal_length)
    power = 2 ** int(math.floor(math.log2(max_allowed)))
    return max(power, 2)

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=22050)
    n_fft = _safe_n_fft(len(y), max_n_fft=2048)

    feats = {
        "filename": os.path.basename(path),
        "duration": float(librosa.get_duration(y=y, sr=sr)),
    }

    # Basic energy / spectral features (use n_fft where relevant)
    feats["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
    feats["rms"] = float(np.mean(librosa.feature.rms(y=y)[0]))
    feats["centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)[0]))
    feats["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)[0]))
    feats["bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)[0]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    for i in range(13):
        feats[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    return feats

def extract_all_audio_features(files):
    rows = []
    for f in tqdm(files, desc="Extracting audio features"):
        try:
            rows.append(extract_audio_features(f))
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")
    return pd.DataFrame(rows)


# --------------------------------------------------
# ALIGNMENT (filename join first; else index-based fallback)
# --------------------------------------------------
def normalize_filename(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    # keep just the base name and normalize case; ensure .wav extension exists
    base = os.path.basename(x)
    if not base.lower().endswith(".wav"):
        base = base + ".wav"
    return base

def align_datasets(audio_df, label_df, heart_df):
    # normalize keys
    audio_df = audio_df.copy()
    label_df = label_df.copy()
    heart_df = heart_df.copy()

    audio_df["__key__"] = audio_df["filename"].apply(normalize_filename)
    label_df["__key__"] = label_df["Filename"].apply(normalize_filename)
    heart_df["__key__"] = heart_df["Filename"].apply(normalize_filename)

    # 1) Try filename join (best when the three sources truly refer to the same files)
    merged = (audio_df
              .merge(label_df, left_on="__key__", right_on="__key__", how="inner", suffixes=("", "_label"))
              .merge(heart_df, left_on="__key__", right_on="__key__", how="inner", suffixes=("", "_heart")))

    # 2) If join fails to recover most samples, fall back to index alignment (what you said is OK)
    expected = min(len(audio_df), len(label_df), len(heart_df))
    if len(merged) < max(1, int(0.8 * expected)) and ALIGN_MODE == "auto":
        # sort by filename (stable) then concatenate by index
        audio_df = audio_df.sort_values("__key__").reset_index(drop=True)
        label_df = label_df.sort_values("__key__").reset_index(drop=True)
        heart_df = heart_df.sort_values("__key__").reset_index(drop=True)

        merged = pd.concat(
            [
                audio_df,
                label_df[["Filename", "Participant", "Label"]].rename(columns={"Filename": "Filename_label"}),
                heart_df.drop(columns=["Filename"]).add_prefix("heart_"),
            ],
            axis=1
        )

    # final tidy: create a unified "label" column (your vocalization meaning)
    if "Label" in merged.columns:
        merged = merged.rename(columns={"Label": "label"})

    # clean up helper cols
    merged = merged.drop(columns=[c for c in merged.columns if c == "__key__"], errors="ignore")
    return merged


# --------------------------------------------------
# MODEL TRAINING (pipeline: numeric + one-hot categorical)
# --------------------------------------------------
def train_multimodal_model(merged_df):
    if TARGET_COL not in merged_df.columns:
        raise ValueError(
            f"TARGET_COL='{TARGET_COL}' not found in merged data columns: {list(merged_df.columns)}"
        )

    # Drop ID-like columns that should not be used as predictive features
    drop_cols = [c for c in ["filename", "Filename", "Filename_label", "Participant"] if c in merged_df.columns]
    df = merged_df.drop(columns=drop_cols, errors="ignore").copy()

    # y (target) and X (features)
    y = df[TARGET_COL].astype(str)  # keep as string for one-hot to handle consistently
    X = df.drop(columns=[TARGET_COL])

    # Split (stratify by target string)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
    ])

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Visuals (pop-ups)
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y)))
    
    # Feature importance (requires feature names from transformer)
    try:
        feature_names = clf.named_steps["preprocess"].get_feature_names_out()
        plot_feature_importance(clf.named_steps["model"], feature_names)
    except Exception:
        # If feature names extraction fails (environment differences), skip gracefully
        print("⚠️ Could not extract expanded feature names for importance plot; skipping.")

    # Save
    joblib.dump({"pipeline": clf, "target_col": TARGET_COL}, MODEL_OUT)
    print(f"\n✅ Model saved to: {MODEL_OUT}")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("\n➡️ Loading labels...")
    label_df = load_audio_labels(AUDIO_LABEL_CSV)
    plot_label_distribution(label_df, "Label")

    print("\n➡️ Loading synthetic heart-rate data...")
    heart_df = load_heart_data(HEART_CSV)
    plot_heart_rate_distribution(heart_df)

    print("\n➡️ Loading WAV files...")
    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    print(f"Found {len(wav_files)} WAV files.")

    if len(wav_files) > 0:
        print("\n📊 Showing visualizations for first audio sample...")
        plot_waveform(wav_files[0])
        plot_melspectrogram(wav_files[0])
        plot_mfcc(wav_files[0])

    print("\n➡️ Extracting audio features...")
    audio_df = extract_all_audio_features(wav_files)

    print("\n➡️ Merging datasets (audio + heartbeat + labels)...")
    merged_df = align_datasets(audio_df, label_df, heart_df)

    print("\n🔎 Merge summary:")
    print("Audio DF:", audio_df.shape)
    print("Label DF:", label_df.shape)
    print("Heart DF:", heart_df.shape)
    print("Merged DF:", merged_df.shape)
    print(merged_df.head())

    merged_df.to_csv(FEATURES_OUT, index=False)
    print(f"\n✅ Saved combined multimodal dataset: {FEATURES_OUT}")

    print("\n➡️ Training multimodal RandomForest model...")
    train_multimodal_model(merged_df)

    print("\n🎉 Done — training + evaluation complete.")

if __name__ == "__main__":
    main()
