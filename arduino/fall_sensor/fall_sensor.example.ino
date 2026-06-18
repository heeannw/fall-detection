// ================================================================
// ?숈긽 媛먯? ?쇱꽌 ??Wemos D1 Mini (ESP8266) + MPU-6050
// ?숈긽 媛먯? ??HTTP POST ??Fall Detection ?쒕쾭
// ================================================================

#include <Wire.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

// ?? ?숋툘 ?ш린留?蹂몄씤 ?섍꼍??留욊쾶 ?섏젙 ?????????????????????????
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://192.168.137.1:8000/arduino/status";
// ?????????????????????????????????????????????????????????????

// 遺? ?
const int BUZZER_PIN = 14; // D5 = GPIO14

// MPU-6050 I2C 二쇱냼
const int MPU_ADDR = 0x68;

// ?숈긽 ?먯젙 ?뚮씪誘명꽣
const float FALL_THRESHOLD  = 1.5;   // g ?댁긽 ??異⑷꺽 (?숈긽 ?섏떖)
const float LYING_THRESHOLD = 0.8;   // g ?댄븯 ???꾩슫 ?곹깭 (?숈긽 ?뺤젙)
const int   CONFIRM_MS      = 800;   // 異⑷꺽 ?????쒓컙(ms) ?덉뿉 ?꾩슫 ?곹깭硫??숈긽
const int   COOLDOWN_MS     = 10000; // ?숈긽 ??踰?蹂대궦 ???湲??쒓컙
const int   NORMAL_INTERVAL = 5000;  // ?뺤긽 ?좏샇 ?꾩넚 二쇨린 (ms)

// ?대? ?곹깭
bool impactDetected = false;
unsigned long impactTime     = 0;
unsigned long lastNormalSent = 0;
unsigned long lastFallSent   = 0;

// ?? MPU-6050 珥덇린????????????????????????????????????????????
void initMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // sleep ?댁젣
  Wire.endTransmission(true);

  // 媛?띾룄 踰붿쐞 짹2g (湲곕낯媛? 蹂寃?遺덊븘??
}

// ?? MPU-6050 媛?띾룄 ?쎄린 (?⑥쐞: g) ??????????????????????????
void readAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();

  // 짹2g 踰붿쐞 ??16384 LSB/g
  ax = rawX / 16384.0;
  ay = rawY / 16384.0;
  az = rawZ / 16384.0;
}

// ?? 遺? 吏곸젒 PWM (ESP8266 tone() WiFi 異⑸룎 ?고쉶) ???????????
void buzzerBeep(int freq, int durationMs) {
  int halfPeriod = 500000 / freq; // 留덉씠?щ줈珥?  unsigned long end = millis() + durationMs;
  while (millis() < end) {
    digitalWrite(BUZZER_PIN, HIGH);
    delayMicroseconds(halfPeriod);
    digitalWrite(BUZZER_PIN, LOW);
    delayMicroseconds(halfPeriod);
  }
}

// ?? HTTP POST ?꾩넚 ???????????????????????????????????????????
void postStatus(const char* status) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[wifi] ?곌껐 ?딄?, ?ъ뿰寃??쒕룄...");
    WiFi.reconnect();
    return;
  }

  WiFiClient client;
  HTTPClient http;
  http.begin(client, SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  String body = String("{\"status\":\"") + status + "\"}";
  int code = http.POST(body);

  if (code > 0) {
    Serial.printf("[http] POST %s ??%d\n", status, code);
  } else {
    Serial.printf("[http] POST ?ㅽ뙣: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

// ?? setup ????????????????????????????????????????????????????
void setup() {
  Serial.begin(9600);

  // 遺?
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // SDA=D2(GPIO4), SCL=D1(GPIO5)
  Wire.begin(4, 5);
  initMPU();
  Serial.println("[mpu] MPU-6050 珥덇린???꾨즺");

  // WiFi ?곌껐
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[wifi] ?곌껐 以?);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 30) {
    delay(500);
    Serial.print(".");
    retry++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[wifi] ?곌껐?? " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[wifi] ?곌껐 ?ㅽ뙣 ??怨꾩냽 ?ъ떆?꾪빀?덈떎");
  }
}

// ?? loop ?????????????????????????????????????????????????????
void loop() {
  float ax, ay, az;
  readAccel(ax, ay, az);

  float total = sqrt(ax * ax + ay * ay + az * az);
  Serial.printf("[accel] total=%.2fg  ax=%.2f ay=%.2f az=%.2f\n", total, ax, ay, az);

  unsigned long now = millis();

  // 1) 異⑷꺽 媛먯? (?꾩쭅 異⑷꺽 異붿쟻 以묒씠 ?꾨땺 ??
  if (!impactDetected && total > FALL_THRESHOLD) {
    Serial.println("[fall] ??異⑷꺽 媛먯?!");
    impactDetected = true;
    impactTime = now;
  }

  // 2) 異⑷꺽 ??CONFIRM_MS ?대궡???꾩슫 ?먯꽭 ?뺤씤 ???숈긽 ?뺤젙
  if (impactDetected) {
    if (now - impactTime < CONFIRM_MS) {
      if (abs(az) < LYING_THRESHOLD) {
        // 荑⑤떎???덉뿉 ?덉쑝硫??ъ쟾??諛⑹?
        if (now - lastFallSent > COOLDOWN_MS) {
          Serial.println("[fall] ?슚 ?숈긽 ?뺤젙! ?쒕쾭 ?꾩넚");
          postStatus("FALL");
          lastFallSent = now;
          // 遺? 3??寃쎄퀬??          for (int i = 0; i < 3; i++) {
            buzzerBeep(1000, 300);
            delay(200);
          }
        }
        impactDetected = false;
      }
    } else {
      // ?쒓컙 珥덇낵 ??異⑷꺽?댁뿀吏留??숈긽 ?꾨떂
      Serial.println("[fall] 異⑷꺽?댁?留??숈긽 ?꾨떂 (?꾩슫 ?먯꽭 誘명솗??");
      impactDetected = false;
    }
  }

  // 3) ?뺤긽 ?좏샇 二쇨린???꾩넚 (?숈긽 荑⑤떎??以묒씠 ?꾨땺 ??
  if (!impactDetected &&
      (now - lastFallSent > COOLDOWN_MS) &&
      (now - lastNormalSent > NORMAL_INTERVAL)) {
    postStatus("NORMAL");
    lastNormalSent = now;
  }

  delay(100); // 10Hz ?섑뵆留?}

