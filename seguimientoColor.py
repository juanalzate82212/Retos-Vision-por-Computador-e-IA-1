import cv2
import numpy as np

#captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

#definición de rangos BGR
# ROJO (BGR)
rojo_bajo = np.array([0, 0, 150])
rojo_alto = np.array([100, 100, 255])

# VERDE
verde_bajo = np.array([40, 130, 40])
verde_alto = np.array([130, 255, 130])

# AZUL
azul_bajo = np.array([130, 30, 40])
azul_alto = np.array([255, 130, 130])

# Kernel para limpiar ruido
kernel = np.ones((5, 5), np.uint8)

#Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #máscaras por color

    mask_rojo = cv2.inRange(frame, rojo_bajo, rojo_alto)
    mask_verde = cv2.inRange(frame, verde_bajo, verde_alto)
    mask_azul = cv2.inRange(frame, azul_bajo, azul_alto)


    #limpieza de ruido
    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_OPEN, kernel)
    mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel)
    mask_azul = cv2.morphologyEx(mask_azul, cv2.MORPH_OPEN, kernel)

    #función para detectar y dibujar
    def detectar_color(mask, color_nombre, color_bgr):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filtrar ruido
                x, y, w, h = cv2.boundingRect(cnt)

                # Dibujar rectángulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

                # Escribir texto
                cv2.putText(
                    frame,
                    color_nombre,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_bgr,
                    2
                )

    #detección de cada color
    detectar_color(mask_rojo, "ROJO", (0, 0, 255))
    detectar_color(mask_verde, "VERDE", (0, 255, 0))
    detectar_color(mask_azul, "AZUL", (255, 0, 0))

    #mostrar resultado
    cv2.imshow("Deteccion de Colores", frame)
    #cv2.imshow("Mascara Rojo", mask_rojo)
    #cv2.imshow("Mascara Verde", mask_verde)
    #cv2.imshow("Mascara Azul", mask_azul)

    if cv2.waitKey(1) & 0xFF == 27:
        break

#Liberar recursos
cap.release()
cv2.destroyAllWindows()