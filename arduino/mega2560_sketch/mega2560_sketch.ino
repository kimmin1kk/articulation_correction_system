//핀 위치
const int firstPin = A0;
const int secondPin = A1;
const int measurementButtonPin = 24;
const int saveButtonPin = 30;
const int buzzerPin = 22;
const int lightPin = 28;

//설정용
int state = 0;
unsigned long startTime = 0;
boolean loopRunning = true;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("--------Arduino ON--------");
  pinMode(measurementButtonPin, INPUT);
  pinMode(saveButtonPin, INPUT);
  pinMode(lightPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
}

void loop() {
    int measurementButton = digitalRead(measurementButtonPin);
    int saveButton = digitalRead(saveButtonPin);
    if(saveButton == HIGH){
      Serial.println("Stop");
      saveButton = LOW;
      delay(1000);

      
    }
    if(measurementButton == HIGH){

      if(state == 1){
        unsigned long endTime = millis();
        Serial.print("End measurement. (");
        Serial.print((double)(endTime - startTime) / 1000 );
        Serial.println("sec)");
        state = 0;
        digitalWrite(lightPin, LOW);
        delay(300);
      }else{
        state = 1;
        Serial.println("Start measurement");
        digitalWrite(lightPin, HIGH);
        delay(300);
        startTime = millis();
      }
    }
    if(state == 1){
      int value1 =  analogRead(firstPin);
      int value2 =  analogRead(secondPin);
      
      if(value1 >= 5) {
        Serial.print("pin1: ");
        Serial.println(value1);
        //tone(buzzerPin, 440, 50);
      }
      if(value2 >= 5) {
        Serial.print("pin2: ");
        Serial.println(value2);
        //tone(buzzerPin, 440, 50);
      }

      delay(10);
    }
  // put your main code here, to run repeatedly:
  
}
