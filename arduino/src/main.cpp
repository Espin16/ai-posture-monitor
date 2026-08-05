#include <Arduino.h>

#define RED_PIN 9
#define GREEN_PIN 10
#define BLUE_PIN 11

const int red[3] = {255, 0, 0};
const int green[3] = {0, 255, 0};
const int blue[3] = {0, 0, 255};

void setup() {
    Serial.begin(9600);
    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(BLUE_PIN, OUTPUT);
}

void loop() {
    analogWrite(RED_PIN, 255);
    delay(1000);
    analogWrite(RED_PIN, 0);
    delay(1000);
}