#include <LedControl.h>

// Definindo os pinos para o controle da matriz 8x8
#define DATA_IN   3
#define LOAD       5
#define CLOCK      4

// Número de módulos 8x8 conectados
#define NUM_DEVICES 1

// Criando o objeto LedControl
LedControl lc = LedControl(DATA_IN, CLOCK, LOAD, NUM_DEVICES);

void setup() {
  // Inicializa todos os dispositivos na matriz de LED
  for (int i = 0; i < NUM_DEVICES; i++) {
    lc.shutdown(i, false);    // A matriz de LEDs está ligada
    lc.setIntensity(i, 8);     // Define o brilho (0-15)
    lc.clearDisplay(i);        // Limpa a matriz de LEDs
  }
}

void loop() {

  wavePattern();
  delay(50);

}

void wavePattern() {
  // Padrão de onda: LEDs acendem em forma de onda, subindo e descendo
  for (int i = 0; i < 8; i++) {
    lc.setRow(0, i, 0xFF);       // Liga todos os LEDs da linha
    delay(100);
    lc.setRow(0, i, 0x00);       // Desliga todos os LEDs da linha
  }

  for (int i = 6; i >= 0; i--) {
    lc.setRow(0, i, 0xFF);       // Liga todos os LEDs da linha
    delay(100);
    lc.setRow(0, i, 0x00);       // Desliga todos os LEDs da linha
  }
}

