#include <Arduino.h>

#define RED_PIN 9
#define GREEN_PIN 10
#define BLUE_PIN 11

const int red[3] = {255, 0, 0};
const int green[3] = {0, 255, 0};
const int blue[3] = {0, 0, 255};



void setup() {
    Serial.begin(9600);
    Serial.setTimeout(1);
    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(BLUE_PIN, OUTPUT);
}

void loop() {
    while (!Serial.available()) {

    }

    delay(50);

    String state = Serial.readString();
    Serial.println(state);
    delay(50);
    state.trim();

    if (state == "G") {
        digitalWrite(RED_PIN, LOW);
        digitalWrite(GREEN_PIN, HIGH);
        digitalWrite(BLUE_PIN, LOW);
    } else if (state == "A") {
        digitalWrite(RED_PIN, HIGH);
        digitalWrite(GREEN_PIN, LOW);
        digitalWrite(BLUE_PIN, LOW);
    } else if (state == "S") {
        digitalWrite(RED_PIN, LOW);
        digitalWrite(GREEN_PIN, LOW);
        digitalWrite(BLUE_PIN, HIGH);
    } else {
        digitalWrite(RED_PIN, LOW);
        digitalWrite(GREEN_PIN, LOW);
        digitalWrite(BLUE_PIN, LOW);
    }


}