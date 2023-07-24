// Motor A connections
int enA = 9;
int in1 = 8;
int in2 = 7;

// LED connection
int led = 12;

// Motor B connections
//int enB = 3;
//int in3 = 5;
//int in4 = 4;

void setup() {
  // Set all the motor control pins to outputs
  pinMode(2, OUTPUT);
  pinMode(led, OUTPUT);
  
  // Turn off motors - Initial state
  digitalWrite(2, LOW);

}

void loop() {


      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(led, LOW);
      analogWrite(enA, 255);
      delay(70);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      delay(100);
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      delay(40);
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      delay(100);
  
}
