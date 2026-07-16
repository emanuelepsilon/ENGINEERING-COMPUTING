#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define DHTPIN 4
#define DHTTYPE DHT11

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

DHT dht(DHTPIN, DHTTYPE);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void drawThermometerIcon(int x, int y) {
  display.drawRoundRect(x + 3, y, 4, 12, 2, SSD1306_WHITE);
  display.fillRect(x + 4, y + 5, 2, 6, SSD1306_WHITE);
  display.fillCircle(x + 5, y + 14, 4, SSD1306_WHITE);
}

void drawDropletIcon(int x, int y) {
  display.fillTriangle(x + 4, y, x, y + 7, x + 8, y + 7, SSD1306_WHITE);
  display.fillCircle(x + 4, y + 9, 4, SSD1306_WHITE);
}

void showStartupScreen() {
  display.clearDisplay();

  display.drawRoundRect(0, 0, 128, 64, 5, SSD1306_WHITE);

  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(24, 18);
  display.println("Emanuel's");

  display.setTextSize(2);
  display.setCursor(34, 32);
  display.println("Room");

  display.display();
  delay(1800);
}

void showErrorScreen() {
  display.clearDisplay();

  display.drawRoundRect(0, 0, 128, 64, 5, SSD1306_WHITE);

  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(28, 14);
  display.println("Sensor error");

  display.drawLine(15, 28, 113, 28, SSD1306_WHITE);

  display.setCursor(18, 39);
  display.println("Check DHT11 wires");

  display.display();
}

void showMainScreen(float temperature, float humidity) {
  display.clearDisplay();

  // Outer frame
  display.drawRoundRect(0, 0, 128, 64, 5, SSD1306_WHITE);

  // Header
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(25, 4);
  display.println("Emanuel's Room");

  display.drawLine(5, 15, 123, 15, SSD1306_WHITE);

  // Temperature side
  drawThermometerIcon(7, 27);

  display.setTextSize(1);
  display.setCursor(21, 23);
  display.println("TEMP");

  display.setCursor(21, 38);
  display.print(temperature, 1);
  display.print(" C");

  // Divider
  display.drawLine(68, 19, 68, 60, SSD1306_WHITE);

  // Humidity side
  drawDropletIcon(76, 28);

  display.setTextSize(1);
  display.setCursor(91, 23);
  display.println("HUM");

  display.setCursor(91, 38);
  display.print(humidity, 0);
  display.print(" %");

  // Small bottom detail line
  display.drawLine(8, 55, 120, 55, SSD1306_WHITE);
  display.setCursor(33, 56);
  display.print("Live room data");

  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();

  Wire.begin(21, 22); // SDA, SCL

  Serial.println("Starting Emanuel's Room");

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED screen not found");
    while (true);
  }

  showStartupScreen();
}

void loop() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT11");
    showErrorScreen();
    delay(2000);
    return;
  }

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print(" C, Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  showMainScreen(temperature, humidity);

  delay(2000);
}