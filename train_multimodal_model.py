#!/usr/bin/env python3

import os
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------------------------------------
# CONFIGURATION — Update only paths
# --------------------------------------------------
AUDIO_DIR = r"C:/Users/user/Desktop/voice/"
AUDIO_LABEL_CSV = r"C:/Users/user/Desktop/voice/dataset_file_directory.csv"
HEART_CSV = r"C:/Users/user/Desktop/voice/synthetic_heart_rate_dataset.csv"

MODEL_OUT     = "multimodal_rf_model.joblib"
FEATURES_OUT  = "multimodal_features.csv"
# --------------------------------------------------


# --------------------------------------------------
# VISUALIZATION FUNCTIONS
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
    librosa.display.specshow(logS, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram: {os.path.basename(path)}")
    plt.tight_layout()
    plt.show()


def plot_mfcc(path):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
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


def plot_label_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df["Label"], order=df["Label"].value_counts().index)
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
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
# 1. Load datasets
# --------------------------------------------------
def load_audio_labels(csv_path):
    df = pd.read_csv(csv_path)
    plot_label_distribution(df)
    return df[["Filename", "Participant", "Label"]]


def load_heart_data(csv_path):
    df = pd.read_csv(csv_path)
    plot_heart_rate_distribution(df)
    return df


# --------------------------------------------------
# 2. Extract audio features (fixed for Librosa ≥ 0.10)
# --------------------------------------------------
def extract_audio_features(path):
    y, sr = librosa.load(path, sr=22050)

    feats = {}
    feats["filename"] = os.path.basename(path)
    feats["duration"] = librosa.get_duration(y=y, sr=sr)

    feats["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
    feats["rms"] = float(np.mean(librosa.feature.rms(y=y)[0]))
    feats["centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    feats["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    feats["bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    return feats


def extract_all_audio_features(files):
    rows = []
    for f in tqdm(files, desc="Extracting audio features"):
        try:
            feats = extract_audio_features(f)
            rows.append(feats)
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")
    return pd.DataFrame(rows)


# --------------------------------------------------
# 3. Merge datasets
# --------------------------------------------------
def merge_datasets(audio_df, label_df, heart_df):
    merged = pd.merge(audio_df, label_df,
                      left_on="filename", right_on="Filename", how="inner")
    merged = pd.merge(merged, heart_df, on="Filename", how="inner")
    merged = merged.rename(columns={"Label": "label"})
    return merged


# --------------------------------------------------
# 4. Train model
# --------------------------------------------------
def train_multimodal_model(df):
    drop_cols = ["filename", "Filename", "Participant"]
    X = df.drop(columns=drop_cols + ["label"], errors="ignore")
    y = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n🎯 Accuracy:", accuracy_score(y_test, preds))
    print("\n📊 Classification Report:\n", classification_report(y_test, preds))

    # POP-UP: Confusion Matrix
    plot_confusion_matrix(y_test, preds, le.classes_)

    # POP-UP: Feature importance
    plot_feature_importance(model, X.columns)

    joblib.dump({"model": model, "label_encoder": le}, MODEL_OUT)
    print(f"\n✅ Model saved to: {MODEL_OUT}")

    return model, le


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("\n➡️ Loading labels...")
    label_df = load_audio_labels(AUDIO_LABEL_CSV)

    print("\n➡️ Loading synthetic heart-rate data...")
    heart_df = load_heart_data(HEART_CSV)

    print("\n➡️ Loading WAV files...")
    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    print(f"Found {len(wav_files)} WAV files.")

    # POP-UP: Show waveform, spectrogram, mfcc of the FIRST audio file
    if len(wav_files) > 0:
        print("\n📊 Showing visualizations for first audio sample...")
        plot_waveform(wav_files[0])
        plot_melspectrogram(wav_files[0])
        plot_mfcc(wav_files[0])

    print("\n➡️ Extracting audio features...")
    audio_df = extract_all_audio_features(wav_files)

    print("\n➡️ Merging datasets (audio + heartbeat + labels)...")
    merged_df = merge_datasets(audio_df, label_df, heart_df)
    merged_df.to_csv(FEATURES_OUT, index=False)
    print(f"Saved combined multimodal dataset: {FEATURES_OUT}")

    print("\n➡️ Training multimodal RandomForest model...")
    train_multimodal_model(merged_df)

    print("\n🎉 Training complete! Visualizations displayed. Model ready.")


if __name__ == "__main__":
    main()

