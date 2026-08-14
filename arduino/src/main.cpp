#include <Arduino.h>

#define RED_PIN 9
#define GREEN_PIN 10
#define BLUE_PIN 11

// const int red[3] = {255, 0, 0};
// const int green[3] = {0, 255, 0};
// const int blue[3] = {0, 0, 255};

String state;

void setColor(int r, int g, int b) {
    digitalWrite(RED_PIN, r);
    digitalWrite(GREEN_PIN, g);
    digitalWrite(BLUE_PIN, b);
}


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

    state = Serial.readString();
    state.trim();
    state.toLowerCase();

    if (state == "g") {
        setColor(0, 255, 0);
    } else if (state == "a") {
        setColor(255, 0, 0);
    } else if (state == "s") {
        setColor(0, 0, 255);
    } else if (state == "off") {
        setColor(0, 0, 0);
    }

}


