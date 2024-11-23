import cv2 as cv
import numpy as np

def roi_bottom_half(frame):
    height, width = frame.shape[:2]
    return frame[height // 2:, :]

def roi_bottom_half3(frame):
    height, width = frame.shape[:2]
    return frame[height // 3:, :]

def main():
    video = cv.VideoCapture(0)

    while True:
        ret, frame = video.read()
        
        if not ret:
            break

        frame = roi_bottom_half(frame)
        frame3 = roi_bottom_half3(frame)

        # Exibir o resultado
        cv.imshow("H/2", frame)
        cv.imshow("H/3", frame3)

        # Sair do loop se a tecla 'q' for pressionada
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
