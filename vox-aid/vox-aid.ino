#include <MAX30105.h>
#include <heartRate.h>
#include <spo2_algorithm.h>
#include <Wire.h>
#include <LiquidCrystal.h>

// LCD Setup (RS=11, EN=12, D4=7, D5=8, D6=9, D7=13)
LiquidCrystal lcd(11, 12, 7, 8, 9, 13);

// MAX30102 Sensor
MAX30105 particleSensor;

// Pin Definitions
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

// Global Variables
bool deviceOn = true; // HARD-CODED FOR TESTING
bool alarmActive = false;
String alarmType = "";
int currentEmotion = 0;
unsigned long lastDisplayUpdate = 0;
const unsigned long displayInterval = 4000;
unsigned long feedTimer = 0;
unsigned long diaperTimer = 0;
const unsigned long feedInterval = 30000;   // 30 sec for testing
const unsigned long diaperInterval = 10000; // 10 sec for testing
int displayCycle = 0;

// Heart Rate Monitoring
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
float aveBPM = 0;

// Motion Detection
bool motionDetected = false;

void setup() {
  Serial.begin(115200);
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

  lcd.begin(16, 2);
  lcd.print("System Init...");
  delay(1000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    lcd.clear();
    lcd.print("HR Sensor Error!");
    while (1);
  }

  particleSensor.setup(25, 4, 2, 100, 411, 4096); // sample setup
  lcd.clear();
  lcd.print("System Ready!");
  delay(1000);

  feedTimer = millis();
  diaperTimer = millis() + 10000;
  lastDisplayUpdate = millis();
}

void loop() {
  if (!deviceOn) {
    digitalWrite(LED_POWER, LOW);
    return;
  }

  // Device ON
  digitalWrite(LED_POWER, HIGH);

  // Read sensors
  readHeartRate();
  motionDetected = digitalRead(PIR_SENSOR);

  // Detect emotion based on BPM and mic
  detectEmotion();

  // Handle routine alarms
  handleFeeding();
  handleDiaper();

  // Reset Buttons
  handleButtonResets();

  // Update LCD every displayInterval
  updateDisplay();
}

// --------------------- HELPER FUNCTIONS ---------------------

void readHeartRate() {
  long irValue = particleSensor.getIR();
  Serial.println(irValue);
  
  if (irValue > 10000) { // Lowered threshold for better detection
    if (checkForBeat(irValue)) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);
      
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        aveBPM = 0;
        for (byte x = 0; x < RATE_SIZE; x++) aveBPM += rates[x];
        aveBPM /= RATE_SIZE;
      }
    }
  }

  Serial.print("BPM: "); Serial.println(beatsPerMinute);
}

void detectEmotion() {
  int micValue = analogRead(MICROPHONE_PIN);

  // Reset LEDs
  digitalWrite(LED_HAPPY, LOW);
  digitalWrite(LED_CALM, LOW);
  digitalWrite(LED_SAD, LOW);

  // Determine emotion
  if (beatsPerMinute > 125 || micValue > 800) {
    currentEmotion = 2; // SAD
    digitalWrite(LED_SAD, HIGH);
  } else if (beatsPerMinute > 100) {
    currentEmotion = 1; // HAPPY
    digitalWrite(LED_HAPPY, HIGH);
  } else {
    currentEmotion = 0; // CALM
    digitalWrite(LED_CALM, HIGH);
  }
}

void handleFeeding() {
  if (millis() - feedTimer >= feedInterval && !alarmActive) {
    alarmActive = true;
    alarmType = "FEED";
    tone(BUZZER_PIN, 900);
  }
}

void handleDiaper() {
  if (millis() - diaperTimer >= diaperInterval && !alarmActive) {
    alarmActive = true;
    alarmType = "DIAPER";
    tone(BUZZER_PIN, 1200);
  }
}

void handleButtonResets() {
  // Feed reset
  if (digitalRead(BTN_FEED_RESET) == LOW && alarmType == "FEED") {
    delay(20);
    if (digitalRead(BTN_FEED_RESET) == LOW) {
      alarmActive = false;
      alarmType = "";
      feedTimer = millis();
      noTone(BUZZER_PIN);
      delay(200);
    }
  }

  // Diaper reset
  if (digitalRead(BTN_DIAPER_RESET) == LOW && alarmType == "DIAPER") {
    delay(20);
    if (digitalRead(BTN_DIAPER_RESET) == LOW) {
      alarmActive = false;
      alarmType = "";
      diaperTimer = millis();
      noTone(BUZZER_PIN);
      delay(200);
    }
  }

  // Emotion reset
  if (digitalRead(BTN_EMOTION_RESET) == LOW) {
    delay(20);
    if (digitalRead(BTN_EMOTION_RESET) == LOW) {
      emotionReset();
      delay(200);
    }
  }
}

void updateDisplay() {
  if (millis() - lastDisplayUpdate >= displayInterval) {
    lcd.clear();

    // Line 0: Alarm or Emotion Status
    lcd.setCursor(0, 0);
    if (alarmActive) {
      if (alarmType == "FEED") lcd.print("--- FEED NOW! ---");
      else if (alarmType == "DIAPER") lcd.print("-DIAPER CHANGE!-");
    } else {
      if (currentEmotion == 0) lcd.print("Status: CALM");
      else if (currentEmotion == 1) lcd.print("Status: HAPPY");
      else if (currentEmotion == 2) lcd.print("Status: SAD");
    }

    // Line 1: Data Cycle
    lcd.setCursor(0, 1);
    if (displayCycle == 0) {
      lcd.print(motionDetected ? "Motion: DETECTED" : "Motion: NONE");
    } else if (displayCycle == 1) {
      lcd.print("BPM: "); lcd.print((int)beatsPerMinute);
    } else if (displayCycle == 2) {
      int micValue = analogRead(MICROPHONE_PIN);
      lcd.print("Mic: "); lcd.print(map(micValue, 0, 1023, 0, 100)); lcd.print("%");
    }

    displayCycle = (displayCycle + 1) % 3;
    lastDisplayUpdate = millis();
  }
}

void emotionReset() {
  digitalWrite(LED_HAPPY, LOW);
  digitalWrite(LED_CALM, LOW);
  digitalWrite(LED_SAD, LOW);
  currentEmotion = 0;
}

void resetAll() {
  alarmActive = false;
  alarmType = "";
  feedTimer = millis();
  diaperTimer = millis();
  noTone(BUZZER_PIN);
  emotionReset();
  digitalWrite(LED_POWER, LOW);
}
