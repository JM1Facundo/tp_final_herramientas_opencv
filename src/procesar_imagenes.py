import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil


def asegurar_dir(path):
    """Crear carpeta si no existe."""
    Path(path).mkdir(parents=True, exist_ok=True)


def clasificar_y_copiar(ruta_raw, ruta_destino):
    """Clasifica imágenes por clase (según nombre del archivo) y copia."""
    asegurar_dir(ruta_destino)

    for img_path in Path(ruta_raw).rglob("*.png"):
        clase = img_path.stem.split("_")[0]
        destino_clase = Path(ruta_destino) / clase
        asegurar_dir(destino_clase)
        shutil.copy(img_path, destino_clase / img_path.name)


def convertir_bn(ruta_in, ruta_out):
    """Convierte imágenes a blanco y negro en OpenCV."""
    asegurar_dir(ruta_out)

    for img_path in Path(ruta_in).rglob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cv2.imwrite(str(Path(ruta_out) / img_path.name), img)


def redimensionar(ruta_in, ruta_out, size=(128, 128)):
    """Redimensiona imágenes usando OpenCV."""
    asegurar_dir(ruta_out)

    for img_path in Path(ruta_in).rglob("*.png"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(Path(ruta_out) / img_path.name), img_resized)


def aplicar_transparencia(ruta_in, ruta_out):
    """Hace blanco → transparente (canal alfa)."""
    asegurar_dir(ruta_out)

    for img_path in Path(ruta_in).rglob("*.png"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        cv2.imwrite(str(Path(ruta_out) / img_path.name), bgra)


def contar_pixeles(ruta_in, ruta_salida_stats):
    """Conteo de píxeles por clase y gráfico final."""
    asegurar_dir(ruta_salida_stats)

    conteo = {}

    for img_path in Path(ruta_in).rglob("*.png"):
        clase = img_path.stem.split("_")[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        total_pixels = img.shape[0] * img.shape[1]
        conteo[clase] = conteo.get(clase, 0) + total_pixels

    clases = sorted(conteo.keys(), key=lambda x: conteo[x])
    valores = [conteo[c] for c in clases]

    plt.figure(figsize=(12, 6))
    plt.bar(clases, valores)
    plt.xticks(rotation=45)
    plt.title("Cantidad de píxeles por clase")
    plt.tight_layout()

    plt.savefig(Path(ruta_salida_stats) / "grafico_pixeles.png")
    plt.close()

    return conteo


def ejecutar_pipeline(paths):
    """Ejecuta todas las fases del procesamiento."""
    print("Fase 1: Clasificación")
    clasificar_y_copiar(paths["raw"], paths["interim_bw"])

    print("Fase 2: Blanco y negro")
    convertir_bn(paths["interim_bw"], paths["interim_resize"])

    print("Fase 3: Redimensionar")
    redimensionar(paths["interim_resize"], paths["interim_transparent"])

    print("Fase 4: Transparencia")
    aplicar_transparencia(paths["interim_transparent"], paths["raw"])  # o a otra carpeta

    print("Fase 5: Conteo de píxeles")
    contar_pixeles(paths["interim_transparent"], paths["processed_stats"])

    print("Pipeline completado.")
