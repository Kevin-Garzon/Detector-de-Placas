import cv2
import pytesseract
import os
from datetime import datetime
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists("placas_detectadas"):
    os.makedirs("placas_detectadas")

cap = cv2.VideoCapture(0)
print("Usando webcam. Presiona 'q' para salir.")

placas_capturadas = []
MAX_CAPTURAS = 5
espera_entre_capturas = 5  # frames entre cada captura para evitar redundancia
contador_frames = 0

def preprocesar_placa(placa_img):
    placa_gray = cv2.cvtColor(placa_img, cv2.COLOR_BGR2GRAY)
    placa_gray = cv2.bilateralFilter(placa_gray, 11, 17, 17)
    _, placa_thresh = cv2.threshold(placa_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return placa_thresh

def obtener_texto(placa_img):
    preprocesada = preprocesar_placa(placa_img)
    return pytesseract.image_to_string(preprocesada, config='--psm 7').strip()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contador_frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 6 and 1000 < cv2.contourArea(cnt) < 15000:
            placa_detectada = frame[y:y+h, x:x+w]

            if contador_frames >= espera_entre_capturas and len(placas_capturadas) < MAX_CAPTURAS:
                placas_capturadas.append(placa_detectada.copy())
                print(f"[INFO] Foto #{len(placas_capturadas)} capturada.")

                contador_frames = 0  # reiniciar contador de espera

            # Dibujar rectángulo solo para visualización
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break  # procesar una placa por frame

    cv2.imshow("Detector de placas (presiona 'q' para salir)", frame)

    if len(placas_capturadas) >= MAX_CAPTURAS:
        print("[INFO] Analizando capturas para determinar la mejor placa...")
        mejores_resultados = [(img, obtener_texto(img)) for img in placas_capturadas]
        mejores_resultados = sorted(mejores_resultados, key=lambda x: len(x[1]), reverse=True)

        mejor_placa, mejor_texto = mejores_resultados[0]
        print(f"[RESULTADO] Mejor texto detectado: {mejor_texto}")

        nombre_archivo = f"placas_detectadas/placa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(nombre_archivo, mejor_placa)
        print(f"[INFO] Imagen guardada: {nombre_archivo}")

        placas_capturadas.clear()  # reiniciar para nuevas detecciones

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()