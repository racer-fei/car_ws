import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from yolov4.tf import YOLOv4

IMG_WIDTH, IMG_HEIGHT = 416, 416
IMG_CHANNELS = 3

# Caminhos de dados
TRAIN_PATH = "/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/frames"
TRAIN_LABELS_PATH = '/Users/sofialinheira/Desktop/IC/tusimple_preprocessed/training/labels'  # Pasta de labels em formato YOLO

# Carregar imagens e labels
def load_yolo_data(image_path, label_path):
    images, labels = [], []
    image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_full_path = os.path.join(image_path, image_file)
        image = cv2.imread(image_full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        images.append(image / 255.0)  # Normalizar

        # Carregar o arquivo de labels correspondente
        label_file = os.path.join(label_path, os.path.splitext(image_file)[0] + '.txt')
        with open(label_file, 'r') as file:
            boxes = []
            for line in file.readlines():
                values = line.strip().split()
                label_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
                boxes.append([label_id, x_center, y_center, width, height])
            labels.append(boxes)

    return np.array(images), np.array(labels)

X_train, Y_train = load_yolo_data(TRAIN_PATH, TRAIN_LABELS_PATH)

# Configuração do YOLOv4
yolo = YOLOv4(tiny=True)
yolo.classes = "/path/to/your/classes.names"  # Arquivo .names com nomes das classes
yolo.input_size = (IMG_WIDTH, IMG_HEIGHT)
yolo.batch_size = 16

# Compilar e configurar o modelo
yolo.make_model()
yolo.load_weights("/path/to/yolov4-tiny.weights", weights_type="yolo")

# Treinamento do YOLOv4
train_dataset = yolo.load_dataset(TRAIN_PATH, TRAIN_LABELS_PATH, "train")  # Ajustar conforme o gerador de dados esperado
val_dataset = yolo.load_dataset(TRAIN_PATH, TRAIN_LABELS_PATH, "val")

yolo.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='binary_crossentropy', metrics=['accuracy'])

history = yolo.model.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=50,  # Definir número de épocas
                         batch_size=yolo.batch_size)

# Salvar o modelo treinado
yolo.model.save('/Users/sofialinheira/Desktop/IC/codigos_teste/network_results/yolo_network.h5')

# Plotar curvas de aprendizado
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento', color='blue')
    plt.plot(history.history['val_loss'], label='Perda de Validação', color='orange')
    plt.title('Curva de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento', color='green')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação', color='red')
    plt.title('Curva de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_learning_curves(history)
