import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from std_msgs.msg import Float32
from cv_bridge import CvBridge

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()

        # Publica o erro de navegação
        self.error_pub = self.create_publisher(Float32, 'lane/error', 10)

        # Configuração da webcam
        self.cap = cv.VideoCapture(0)  # Abre a webcam (0 é geralmente a primeira webcam)
        if not self.cap.isOpened():
            self.get_logger().error("Não foi possível acessar a webcam!")
            return
        # Mensagem de sucesso ao iniciar o nó
        self.get_logger().info("Nó 'perception_node' iniciado com sucesso!")

    def crop_left_half(self, frame):
        height, width = frame.shape[:2]
        return frame[:, :width // 2]

    def roi_bottom_half(self, frame):
        height, width = frame.shape[:2]
        return frame[height // 2:, :]

    def calculate_angle(self, x1, y1, x2, y2):
        delta_y = y2 - y1
        delta_x = x2 - x1
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle

    def segment_color(self, frame):
        # Define as faixas de cor para a pista (acinzentada) e as delimitações (azul e branco)
        # Pista: Faixa de cores acinzentadas
        lower_gray = np.array([26, 30, 41])  # Cor mínima para a pista
        upper_gray = np.array([183, 171, 153])  # Cor máxima para a pista
        
        # Azul: Faixa de cores azuis delimitadoras
        lower_blue = np.array([27, 73, 117])  # Cor mínima para azul
        upper_blue = np.array([134, 182, 209])  # Cor máxima para azul
        
        # Branco: Faixa de cores brancas delimitadoras
        lower_white = np.array([118, 140, 169])  # Cor mínima para branco
        upper_white = np.array([226, 222, 214])  # Cor máxima para branco

        # Converte a imagem para o espaço de cor HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Segmentação das cores
        mask_gray = cv.inRange(hsv_frame, lower_gray, upper_gray)
        mask_blue = cv.inRange(hsv_frame, lower_blue, upper_blue)
        mask_white = cv.inRange(hsv_frame, lower_white, upper_white)

        # Combina as máscaras para identificar a pista
        mask = mask_gray  # A pista será a área com cor acinzentada
        return mask, mask_blue, mask_white

    def process_frame(self, frame):
        # Pré-processamento da imagem com segmentação por cor
        mask, mask_blue, mask_white = self.segment_color(frame)

        # Aplica a máscara na imagem para isolar a área da pista
        masked_frame = cv.bitwise_and(frame, frame, mask=mask)

        # Usando a máscara para limitar a área onde as linhas serão detectadas
        frame = self.crop_left_half(masked_frame)
        frame = self.roi_bottom_half(frame)
        frame_copy = frame.copy()

        # Conversão para escala de cinza para detectar bordas
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (7, 7), 0)
        edges = cv.Canny(blurred, 50, 150)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv.dilate(edges, kernel, iterations=1)
        edges = cv.erode(edges, kernel, iterations=1)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=5)

        if lines is not None:
            left_lines, right_lines = [], []
            left_angles, right_angles = [], []
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = self.calculate_angle(x1, y1, x2, y2)
                if abs(angle) > 5:  # Ignora ângulos muito pequenos
                    if x1 < frame.shape[1] // 2:  # Linha à esquerda
                        left_lines.append((x1, y1, x2, y2))
                        left_angles.append(angle)
                    else:  # Linha à direita
                        right_lines.append((x1, y1, x2, y2))
                        right_angles.append(angle)

            # Desenha as linhas detectadas na imagem original (copiada)
            for x1, y1, x2, y2 in left_lines + right_lines:
                cv.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Linhas verdes

            if left_lines and right_lines:
                # Calcular os centros das linhas esquerda e direita
                left_center = np.mean([x1 for x1, _, _, _ in left_lines])
                right_center = np.mean([x2 for _, _, x2, _ in right_lines])
                center_line = (left_center + right_center) / 2

                # Calcular a largura da faixa
                lane_width = right_center - left_center

                # Calcular o ângulo médio das linhas esquerda e direita
                avg_left_angle = np.mean(left_angles) if left_angles else 0
                avg_right_angle = np.mean(right_angles) if right_angles else 0
                avg_angle = (avg_left_angle + avg_right_angle) / 2

                # Ajustar o erro com base no ângulo
                angle_factor = 0.1  # Fator de ajuste do erro baseado no ângulo
                angle_adjustment = angle_factor * avg_angle

                # Erro de navegação considerando a posição central e o ângulo
                error = center_line - (frame.shape[1] / 2) + angle_adjustment

                # Considerar a largura da faixa no erro
                lane_width_factor = 0.1  # Fator de ajuste baseado na largura da faixa
                width_adjustment = lane_width_factor * (frame.shape[1] - lane_width)

                # Combinar erro baseado na posição, ângulo e largura da faixa
                final_error = error + width_adjustment

                # Publica o erro de navegação
                self.error_pub.publish(Float32(data=final_error))
                self.get_logger().info(f"Erro publicado: {final_error}")

        cv.imwrite("/home/roboracer/imgs/frame_copy.jpg", frame_copy)

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()  # Lê o próximo frame da webcam
            if not ret:
                self.get_logger().error("Falha ao capturar frame da webcam!")
                break

            self.process_frame(frame)  # Processa o frame

    def stop(self):
        self.cap.release()  # Libera a câmera quando o programa terminar

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()

    node.run()

    # Libera recursos ao encerrar
    node.stop()
    node.destroy_node()
    rclpy.shutdown()
    cv.destroyAllWindows()  # Fecha todas as janelas OpenCV ao encerrar

if __name__ == '__main__':
    main()
