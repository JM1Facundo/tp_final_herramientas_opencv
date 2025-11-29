import cv2
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

# 1. Creamos imagenes usando OpenCV:
def cargar_imagen(ruta):
    """Carga una imagen usando OpenCV y la retorna en BGR."""
    img = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    return img

# 2. Redimensionamos la imagenes:
def redimensionar(img, size=(800, 800)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# 3. Conversion a escala de grises:
def a_grises(img):
    """Retorna imagen en escala de grises."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Recorte automatico:
def recortar_bordes(img):
    """Recorta los bordes blancos basándose en píxeles no vacíos."""
    coords = np.column_stack(np.where(img < 255))

    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img[y_min:y_max+1, x_min:x_max+1]

# 5. Generar PNG con transparencias:
def agregar_transparencia(img_gris):
    """Convierte fondo blanco en transparencia usando canal alpha."""
    rgba = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2BGRA)

    # Fondo blanco → alpha = 0
    rgba[:, :, 3] = np.where(img_gris == 255, 0, 255)

    return rgba

# 6. Colorear figuras:
def colorizar(img_rgba, hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    mask = img_rgba[:, :, 3] == 255
    img_rgba[mask, :3] = rgb

    return img_rgba

# 7. Dectectar puntos extremos:
def puntos_extremos(img_binaria):
    contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return None

    c = max(contornos, key=cv2.contourArea)

    izquierda = tuple(c[c[:, :, 0].argmin()][0])
    derecha   = tuple(c[c[:, :, 0].argmax()][0])
    arriba    = tuple(c[c[:, :, 1].argmin()][0])
    abajo     = tuple(c[c[:, :, 1].argmax()][0])

    return izquierda, derecha, arriba, abajo

# 8. Escribir nombre centrado:
def escribir_nombre(img_rgba, texto):
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
    h, w, _ = img_bgr.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2

    text_w, text_h = cv2.getTextSize(texto, font, scale, thickness)[0]
    x = (w - text_w) // 2
    y = h - 20

    cv2.putText(img_bgr, texto, (x, y), font, scale, (0, 0, 0), thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

# 9. Contar pixeles.
def contar_pixeles(mask):
    return np.sum(mask == 255)

# 10. Mostrar imagenes:
def mostrar(img, title=""):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
