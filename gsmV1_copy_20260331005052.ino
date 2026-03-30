#include <MAX30105.h>
#include <heartRate.h>
#include <Wire.h>
#include <LiquidCrystal.h>
#include <SoftwareSerial.h>

// ============================================
// GSM SETUP
// ============================================
SoftwareSerial sim800(A8, A9); // RX, TX
String phoneNumber = "+256763318697"; // replace with your number
unsigned long lastSMS = 0;
const unsigned long SMS_INTERVAL = 15000; // 15 sec

// ============================================
// ALARM SCHEDULER CLASS (unchanged - 1 min / 2.5 min)
// ============================================
enum AlarmType {
  ALARM_NONE = 0,
  ALARM_DIAPER = 1,
  ALARM_FEEDING = 2
};

struct AlarmEvent {
  AlarmType type;
  unsigned long lastTriggerTime;
  unsigned long intervalMs;
  unsigned long durationMs;
  unsigned long alarmStartTime;
  bool isActive;
};

class AlarmScheduler {
private:
  AlarmEvent diaperAlarm;
  AlarmEvent feedingAlarm;
  int buzzerPin;
  int diaperTone;
  int feedingTone;

public:
  AlarmScheduler(int pin) : buzzerPin(pin), diaperTone(1200), feedingTone(900) {
    diaperAlarm.type = ALARM_DIAPER;
    diaperAlarm.lastTriggerTime = 0;
    diaperAlarm.intervalMs = 60000;
    diaperAlarm.durationMs = 2000;
    diaperAlarm.alarmStartTime = 0;
    diaperAlarm.isActive = false;

    feedingAlarm.type = ALARM_FEEDING;
    feedingAlarm.lastTriggerTime = 0;
    feedingAlarm.intervalMs = 150000;
    feedingAlarm.durationMs = 2000;
    feedingAlarm.alarmStartTime = 0;
    feedingAlarm.isActive = false;

    pinMode(buzzerPin, OUTPUT);
    noTone(buzzerPin);
  }

  void begin() {
    unsigned long now = millis();
    diaperAlarm.lastTriggerTime = now;
    feedingAlarm.lastTriggerTime = now;
  }

  void update() {
    unsigned long now = millis();

    if (!diaperAlarm.isActive && (now - diaperAlarm.lastTriggerTime >= diaperAlarm.intervalMs)) {
      diaperAlarm.isActive = true;
      diaperAlarm.alarmStartTime = now;
      tone(buzzerPin, diaperTone);
      Serial.println("[ALARM] DIAPER alarm started (1 min)");
    }

    if (!feedingAlarm.isActive && (now - feedingAlarm.lastTriggerTime >= feedingAlarm.intervalMs)) {
      feedingAlarm.isActive = true;
      feedingAlarm.alarmStartTime = now;
      tone(buzzerPin, feedingTone);
      Serial.println("[ALARM] FEEDING alarm started (2.5 min)");
    }

    if (feedingAlarm.isActive) tone(buzzerPin, feedingTone);
    else if (diaperAlarm.isActive) tone(buzzerPin, diaperTone);

    stopExpiredAlarms(now);
  }

  void stopExpiredAlarms(unsigned long now) {
    bool toneNeedsUpdate = false;

    if (diaperAlarm.isActive && (now - diaperAlarm.alarmStartTime >= diaperAlarm.durationMs)) {
      diaperAlarm.isActive = false;
      diaperAlarm.alarmStartTime = 0;
      diaperAlarm.lastTriggerTime = now;
      toneNeedsUpdate = true;
      Serial.println("[ALARM] DIAPER alarm ended");
    }

    if (feedingAlarm.isActive && (now - feedingAlarm.alarmStartTime >= feedingAlarm.durationMs)) {
      feedingAlarm.isActive = false;
      feedingAlarm.alarmStartTime = 0;
      feedingAlarm.lastTriggerTime = now;
      toneNeedsUpdate = true;
      Serial.println("[ALARM] FEEDING alarm ended");
    }

    if (toneNeedsUpdate) {
      if (feedingAlarm.isActive) tone(buzzerPin, feedingTone);
      else if (diaperAlarm.isActive) tone(buzzerPin, diaperTone);
      else noTone(buzzerPin);
    }
  }

  AlarmType getActiveAlarm() {
    if (feedingAlarm.isActive) return ALARM_FEEDING;
    if (diaperAlarm.isActive) return ALARM_DIAPER;
    return ALARM_NONE;
  }

  unsigned long getAlarmTimeRemaining() {
    unsigned long now = millis();
    if (feedingAlarm.isActive) {
      long remaining = (long)feedingAlarm.durationMs - (long)(now - feedingAlarm.alarmStartTime);
      return (remaining > 0) ? remaining : 0;
    }
    if (diaperAlarm.isActive) {
      long remaining = (long)diaperAlarm.durationMs - (long)(now - diaperAlarm.alarmStartTime);
      return (remaining > 0) ? remaining : 0;
    }
    return 0;
  }

  void silenceAlarm(AlarmType type) {
    if (type == ALARM_DIAPER && diaperAlarm.isActive) {
      diaperAlarm.isActive = false;
      diaperAlarm.alarmStartTime = 0;
      diaperAlarm.lastTriggerTime = millis();
      Serial.println("[BTN] DIAPER alarm silenced");
    }
    if (type == ALARM_FEEDING && feedingAlarm.isActive) {
      feedingAlarm.isActive = false;
      feedingAlarm.alarmStartTime = 0;
      feedingAlarm.lastTriggerTime = millis();
      Serial.println("[BTN] FEEDING alarm silenced");
    }

    if (feedingAlarm.isActive) tone(buzzerPin, feedingTone);
    else if (diaperAlarm.isActive) tone(buzzerPin, diaperTone);
    else noTone(buzzerPin);
  }

  void resetAll() {
    diaperAlarm.isActive = false;
    diaperAlarm.alarmStartTime = 0;
    diaperAlarm.lastTriggerTime = millis();
    feedingAlarm.isActive = false;
    feedingAlarm.alarmStartTime = 0;
    feedingAlarm.lastTriggerTime = millis();
    noTone(buzzerPin);
  }
};
// ============================================
// SOUND CLASSIFIER CLASS (unchanged)
// ============================================
enum SoundType {
  SOUND_SILENCE = 0,
  SOUND_GROAN = 1,
  SOUND_LAUGH = 2,
  SOUND_CRY = 3
};

class SoundClassifier {
private:
  static const int MIC_BUFFER_SIZE = 64;
  int amplitudeBuffer[MIC_BUFFER_SIZE];
  int bufferIndex;
  int micPin;
  unsigned long lastSampleWindow = 0;
  unsigned long lastAnalysisTime = 0;
  int lastAmplitude = 0;
  SoundType lastClassified = SOUND_SILENCE;
  const unsigned long SAMPLE_WINDOW_MS = 50;
  const unsigned long ANALYSIS_INTERVAL = 400;

public:
  SoundClassifier(int pin) : micPin(pin), bufferIndex(0) {
    pinMode(micPin, INPUT);
    for (int i = 0; i < MIC_BUFFER_SIZE; i++) amplitudeBuffer[i] = 0;
  }

  void update() {
    unsigned long now = millis();
    if (now - lastSampleWindow >= SAMPLE_WINDOW_MS) {
      lastSampleWindow = now;
      int signalMin = 1023;
      int signalMax = 0;
      unsigned long start = millis();
      while (millis() - start < SAMPLE_WINDOW_MS) {
        int sample = analogRead(micPin);
        if (sample < signalMin) signalMin = sample;
        if (sample > signalMax) signalMax = sample;
      }
      int peakToPeak = signalMax - signalMin;
      lastAmplitude = peakToPeak;
      amplitudeBuffer[bufferIndex] = peakToPeak;
      bufferIndex = (bufferIndex + 1) % MIC_BUFFER_SIZE;
    }
  }

  SoundType classify() {
    unsigned long now = millis();
    if (now - lastAnalysisTime < ANALYSIS_INTERVAL) return lastClassified;
    lastAnalysisTime = now;

    int sum = 0, maxAmp = 0, highCount = 0, mediumCount = 0;
    for (int i = 0; i < MIC_BUFFER_SIZE; i++) {
      int a = amplitudeBuffer[i];
      sum += a;
      if (a > maxAmp) maxAmp = a;
      if (a > 180) highCount++;
      if (a > 90) mediumCount++;
    }
    int avgAmp = sum / MIC_BUFFER_SIZE;

    if (avgAmp < 20 && maxAmp < 40) lastClassified = SOUND_SILENCE;
    else if (maxAmp > 300 || highCount > 20) lastClassified = SOUND_CRY;
    else if (maxAmp > 180 && mediumCount > 8) lastClassified = SOUND_LAUGH;
    else if (avgAmp > 50) lastClassified = SOUND_GROAN;
    else lastClassified = SOUND_SILENCE;

    return lastClassified;
  }

  int getAmplitudePercent() {
    return constrain(map(lastAmplitude, 0, 500, 0, 100), 0, 100);
  }

  static const char* soundTypeToString(SoundType type) {
    switch (type) {
      case SOUND_SILENCE: return "SILENT";
      case SOUND_GROAN: return "GROAN";
      case SOUND_LAUGH: return "LAUGH";
      case SOUND_CRY: return "CRY";
      default: return "UNKNOWN";
    }
  }
};

// ======================
// DISPLAY MANAGER CLASS
// ============================================
struct SensorReadings {
  float heartRate;
  long irValue;
  bool fingerDetected;
  bool motionDetected;
  int micAmplitude;
  SoundType soundType;
  int currentEmotion;
  AlarmType activeAlarm;
  unsigned long alarmTimeRemaining;
};

class DisplayManager {
private:
  LiquidCrystal lcd;
  unsigned long lastDisplayUpdate;
  const unsigned long DISPLAY_INTERVAL = 3000;
  int displayCycle;

public:
  DisplayManager(int rs, int en, int d4, int d5, int d6, int d7)
    : lcd(rs, en, d4, d5, d6, d7), lastDisplayUpdate(0), displayCycle(0) {
    lcd.begin(16, 2);
    lcd.clear();
    lcd.print("System Init...");
  }

  void begin() {
    lcd.clear();
    lcd.print("System Ready!");
    delay(1000);
    lastDisplayUpdate = millis();
    displayCycle = 0;
  }

  void update(SensorReadings readings) {
    unsigned long now = millis();

    if (readings.activeAlarm != ALARM_NONE) {
      lcd.clear();
      displayAlarmScreen(readings);
      lastDisplayUpdate = now;
      return;
    }

    if (now - lastDisplayUpdate >= DISPLAY_INTERVAL) {
      lcd.clear();
      displayScreen(readings);
      lastDisplayUpdate = now;
      displayCycle = (displayCycle + 1) % 4;
    }
  }

  void displayScreen(SensorReadings readings) {
    switch (displayCycle) {
      case 0: displayEmotionScreen(readings); break;
      case 1: displayMotionScreen(readings); break;
      case 2: displayHeartRateScreen(readings); break;
      case 3: displaySoundScreen(readings); break;
    }
  }

  void displayAlarmScreen(SensorReadings readings) {
    if (readings.activeAlarm == ALARM_FEEDING) lcd.print("FEEDING ALARM!");
    else lcd.print("DIAPER ALARM!");
    lcd.setCursor(0, 1);
    lcd.print("Reset Btn to Stop");
  }

  void displayEmotionScreen(SensorReadings readings) {
    lcd.print("Status: ");
    switch (readings.currentEmotion) {
      case 0: lcd.print("CALM"); break;
      case 1: lcd.print("HAPPY"); break;
      case 2: lcd.print("SAD"); break;
      default: lcd.print("?"); break;
    }
    lcd.setCursor(0, 1);
    lcd.print("No Alarm Active");
  }

  void displayMotionScreen(SensorReadings readings) {
    lcd.print("Motion Status:");
    lcd.setCursor(0, 1);
    lcd.print(readings.motionDetected ? "DETECTED!" : "None (Resting)");
  }

  void displayHeartRateScreen(SensorReadings readings) {
    lcd.print("GYMAX / HR:");
    lcd.setCursor(0, 1);
    if (!readings.fingerDetected) {
      lcd.print("No Finger Detect");
    } else if (readings.heartRate > 0) {
      lcd.print("BPM: ");
      lcd.print((int)readings.heartRate);
    } else {
      lcd.print("Reading...");
    }
  }

  void displaySoundScreen(SensorReadings readings) {
    lcd.print("Sound:");
    lcd.print(SoundClassifier::soundTypeToString(readings.soundType));
    lcd.setCursor(0, 1);
    lcd.print("Lvl:");
    lcd.print(readings.micAmplitude);
    lcd.print("%");
  }

  void showMessage(const char* line0, const char* line1 = "") {
    lcd.clear();
    lcd.print(line0);
    if (line1 && strlen(line1) > 0) {
      lcd.setCursor(0, 1);
      lcd.print(line1);
    }
  }
};


// ============================================
// PIN DEFINITIONS
// ============================================
#define LCD_RS 11
#define LCD_EN 12
#define LCD_D4 7
#define LCD_D5 8
#define LCD_D6 9
#define LCD_D7 13
#define PIR_SENSOR 6
#define MICROPHONE_PIN A0
#define BUZZER_PIN 10
#define LED_POWER 2
#define LED_HAPPY 3
#define LED_CALM 4
#define LED_SAD 5
#define BTN_POWER A1
#define BTN_FEED_RESET A2
#define BTN_DIAPER_RESET A3
#define BTN_EMOTION_RESET A6

// ============================================
// GLOBAL OBJECTS & STATE
// ============================================
MAX30105 particleSensor;
AlarmScheduler alarmScheduler(BUZZER_PIN);
SoundClassifier soundClassifier(MICROPHONE_PIN);
DisplayManager displayManager(LCD_RS,LCD_EN,LCD_D4,LCD_D5,LCD_D6,LCD_D7);

struct GlobalState {
  byte rates[4] = {0,0,0,0};
  byte rateSpot = 0;
  long lastBeat = 0;
  float beatsPerMinute = 0;
  float aveBPM = 0;
  long irValue = 0;
  bool fingerDetected = false;
  bool motionDetected = false;
  int currentEmotion = 0;
  unsigned long lastBtnPressTime = 0;
  const unsigned long DEBOUNCE_DELAY = 300;
  bool deviceOn = true;
};

GlobalState state;

// ============================================
// GSM FUNCTIONS
// ============================================
void initGSM() {
  Serial.println("=== SIM800 Init ===");
  sim800.begin(9600);
  delay(1000);
  sim800.println("AT"); delay(500);
  sim800.println("AT+CPIN?"); delay(500);
  sim800.println("AT+CREG?"); delay(500);
  sim800.println("AT+CSQ"); delay(500);
  sim800.println("AT+CMGF=1"); // text mode
  delay(500);
  Serial.println("✅ GSM Ready!");
}

bool sendSMS(String number, String text) {
  sim800.print("AT+CMGS=\""); sim800.print(number); sim800.println("\"");
  delay(1000);

  String response = "";
  unsigned long start = millis();
  while (millis() - start < 5000) {
    while (sim800.available()) response += (char)sim800.read();
    if (response.indexOf(">") != -1) break;
  }
  if (response.indexOf(">") == -1) {
    Serial.println("❌ SMS prompt '>' not received!");
    return false;
  }

  sim800.print(text);
  delay(500);
  sim800.write(26); // Ctrl+Z
  delay(5000);

  response = "";
  start = millis();
  while (millis() - start < 10000) {
    while (sim800.available()) response += (char)sim800.read();
  }

  if (response.indexOf("OK") != -1 || response.indexOf("+CMGS:") != -1) {
    Serial.println("✅ SMS SENT!");
    return true;
  } else {
    Serial.println("❌ SMS failed!");
    return false;
  }
}

void sendBabySummary() {
  if (millis() - lastSMS < SMS_INTERVAL) return;
  lastSMS = millis();

  String sms = "VOX-AID ID001:\n";
  sms += "BPM: " + String(state.aveBPM, 1) + "\n";
  sms += "Finger: " + String(state.fingerDetected ? "YES" : "NO") + "\n";
  sms += "Motion: " + String(state.motionDetected ? "YES" : "NO") + "\n";
  sms += "Sound Level: " + String(soundClassifier.getAmplitudePercent()) + "%\n";
  sms += "Emotion: ";
  sms += (state.currentEmotion==0?"CALM":state.currentEmotion==1?"HAPPY":"SAD") + String("\n");
  sms += "Alarm: ";
  AlarmType alarm = alarmScheduler.getActiveAlarm();
  if (alarm == ALARM_NONE) sms += "NONE";
  else if (alarm == ALARM_DIAPER) sms += "DIAPER";
  else sms += "FEEDING";

  sendSMS(phoneNumber, sms);
}

// ============================================
// SETUP
// ============================================
void setup() {
  Serial.begin(115200);
  while(!Serial);

  pinMode(LED_POWER, OUTPUT);
  pinMode(LED_HAPPY, OUTPUT);
  pinMode(LED_CALM, OUTPUT);
  pinMode(LED_SAD, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_SENSOR, INPUT);
  pinMode(BTN_POWER, INPUT_PULLUP);
  pinMode(BTN_FEED_RESET, INPUT_PULLUP);
  pinMode(BTN_DIAPER_RESET, INPUT_PULLUP);
  pinMode(BTN_EMOTION_RESET, INPUT_PULLUP);

  displayManager.showMessage("System Init...","");
  Serial.println("[SETUP] Initializing MAX30105...");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERROR] MAX30105 not found!");
    displayManager.showMessage("HR Sensor Error!","Check wiring");
    while(1);
  }

  particleSensor.setup(255,4,2,100,411,16384);
  particleSensor.setPulseAmplitudeRed(0xFF);
  particleSensor.setPulseAmplitudeIR(0xFF);
  particleSensor.setPulseAmplitudeGreen(0);

  alarmScheduler.begin();
  displayManager.begin();
  digitalWrite(LED_POWER,HIGH);

  initGSM(); // initialize SIM800

  Serial.println("[SETUP] Baby Monitor Ready!");
}

// ============================================
// MAIN LOOP
// ============================================
void loop() {
  handleButtons();

  digitalWrite(LED_POWER, state.deviceOn ? HIGH : LOW);

  if (!state.deviceOn) {
    alarmScheduler.resetAll();
    digitalWrite(LED_HAPPY, LOW);
    digitalWrite(LED_CALM, LOW);
    digitalWrite(LED_SAD, LOW);
    displayManager.showMessage("System Paused","Press Power Btn");
    delay(200);
    return;
  }

  readHeartRate();
  state.motionDetected = digitalRead(PIR_SENSOR);
  soundClassifier.update();
  detectEmotion();
  alarmScheduler.update();

  SensorReadings readings;
  readings.heartRate = state.aveBPM;
  readings.irValue = state.irValue;
  readings.fingerDetected = state.fingerDetected;
  readings.motionDetected = state.motionDetected;
  readings.micAmplitude = soundClassifier.getAmplitudePercent();
  readings.soundType = soundClassifier.classify();
  readings.currentEmotion = state.currentEmotion;
  readings.activeAlarm = alarmScheduler.getActiveAlarm();
  readings.alarmTimeRemaining = alarmScheduler.getAlarmTimeRemaining();

  displayManager.update(readings);

  // Send SMS summary every interval
  sendBabySummary();

  delay(500);
}

// ============================================
// HEART RATE READING - TUNED FOR REAL BPM (60-120 range)
// ============================================
void readHeartRate() {
  state.irValue = particleSensor.getIR();

  if (state.irValue < 7000) {
    state.fingerDetected = false;
    state.beatsPerMinute = 0;
    state.aveBPM = 0;
    return;
  }

  state.fingerDetected = true;

  // Library beat detection first
  bool beatDetected = checkForBeat(state.irValue);

  // Custom rising-edge detection - TUNED LOWER for real human pulses
  static long prevIR = 0;
  static bool risingEdge = false;
  long diff = state.irValue - prevIR;
  prevIR = state.irValue;

  if (diff > 500 && !risingEdge) {          // ← LOWERED from 800 to 500 (catches real beats more often)
    risingEdge = true;
    beatDetected = true;
  }
  if (diff < -400) risingEdge = false;

  if (beatDetected) {
    unsigned long now = millis();
    long delta = now - state.lastBeat;

    // Refractory period - ignore fake double triggers
    if (delta < 250) return;

    state.lastBeat = now;

    if (delta > 300 && delta < 2000) {      // realistic human heartbeat range
      state.beatsPerMinute = 60000.0 / delta;

      if (state.beatsPerMinute > 35 && state.beatsPerMinute < 220) {
        state.rates[state.rateSpot++] = (byte)state.beatsPerMinute;
        state.rateSpot %= 4;

        float sum = 0;
        for (byte i = 0; i < 4; i++) sum += state.rates[i];
        state.aveBPM = sum / 4.0;

        Serial.print("→ REAL BEAT! delta=");
        Serial.print(delta);
        Serial.print("ms → BPM=");
        Serial.println(state.aveBPM, 1);
      }
    }
  }
}

// ============================================
// EMOTION DETECTION 
// ============================================
void detectEmotion() {
  digitalWrite(LED_HAPPY, LOW);
  digitalWrite(LED_CALM, LOW);
  digitalWrite(LED_SAD, LOW);

  SoundType sound = soundClassifier.classify();
  int micAmp = soundClassifier.getAmplitudePercent();

  if (sound == SOUND_CRY || state.beatsPerMinute > 125 || micAmp > 75) {
    state.currentEmotion = 2;
    digitalWrite(LED_SAD, HIGH);
  }
  else if (sound == SOUND_LAUGH || state.beatsPerMinute > 100 || state.motionDetected) {
    state.currentEmotion = 1;
    digitalWrite(LED_HAPPY, HIGH);
  }
  else {
    state.currentEmotion = 0;
    digitalWrite(LED_CALM, HIGH);
  }
}

// ============================================
// BUTTON HANDLING (unchanged)
// ============================================
void handleButtons() {
  unsigned long now = millis();
  if (now - state.lastBtnPressTime < state.DEBOUNCE_DELAY) return;

  if (digitalRead(BTN_FEED_RESET) == LOW) {
    state.lastBtnPressTime = now;
    alarmScheduler.silenceAlarm(ALARM_FEEDING);
    return;
  }
  if (digitalRead(BTN_DIAPER_RESET) == LOW) {
    state.lastBtnPressTime = now;
    alarmScheduler.silenceAlarm(ALARM_DIAPER);
    return;
  }
  if (digitalRead(BTN_EMOTION_RESET) == LOW) {
    state.lastBtnPressTime = now;
    state.currentEmotion = 0;
    digitalWrite(LED_HAPPY, LOW);
    digitalWrite(LED_CALM, HIGH);
    digitalWrite(LED_SAD, LOW);
    Serial.println("[BTN] Emotion reset");
    return;
  }
  if (digitalRead(BTN_POWER) == LOW) {
    state.lastBtnPressTime = now;
    state.deviceOn = !state.deviceOn;
    Serial.print("[BTN] Device power: ");
    Serial.println(state.deviceOn ? "ON" : "OFF");
    return;
  }
}