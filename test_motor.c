const int escPin = 9; // Pino de controle PWM

void setup() {
  pinMode(escPin, OUTPUT); // Define o pino como saída
  Serial.begin(9600);
}

void loop() {
  // Acelera o motor gradualmente de 0 a 255 (mínimo para máximo)
  for (int pwmValue = 0; pwmValue <= 255; pwmValue++) {
    analogWrite(escPin, pwmValue);  // Envia o valor PWM para o ESC
    delay(100);  // Atraso de 100ms entre cada aumento
  }

  delay(1000);  // Espera 1 segundo com o motor na velocidade máxima

  // Diminui a velocidade do motor gradualmente de 255 a 0
  for (int pwmValue = 255; pwmValue >= 0; pwmValue--) {
    analogWrite(escPin, pwmValue);  // Envia o valor PWM para o ESC
    delay(100);  // Atraso de 100ms entre cada diminuição
  }

  delay(1000);  // Espera 1 segundo com o motor parado
}
