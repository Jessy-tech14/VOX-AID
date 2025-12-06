# VOX-AID
Vox-Aid is an assistive Arduino based device fro non-verbal Celebral Palsy patients. It detects emotion using heartrate, sound, and motion, then alertd caregivers through LEDs, buzzer, and LCD messages. It reminnds fro feeding and diaper changes, and helps improve patient communication
Audio:
- Description: Vox-Aid uses audio input from non-verbal CP patients to detect emotions like pain, discomfort, or happiness .
- Data Collection: Captured vocalizations/non-verbal sounds via Arduino-based setup.
- Challenge: Had over 7000 audio files but uploaded a subset (very few) due to size constraints . Analyzed patterns like pitch, tone for emotion detection .

ML Visualization:
- Emotion Classification: Used ML models to classify emotions from audio features .
- Visuals: Confusion matrices, accuracy plots show detection effectiveness .

Datasets:
- Audio Samples: Limited upload (subset of 7077) with labels (pain, comfort, etc.) .
- Features Extracted: Pitch, MFCC, spectral features for ML input .

Arduino-Based System:
- Hardware: Arduino captures audio, sends data for processing .
- Goal: Help caregivers detect emotions in physically immobile patients
