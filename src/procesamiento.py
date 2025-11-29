import pathlib
from procesar_imagenes import ejecutar_pipeline

def main():
    # Ruta base del proyecto (carpeta raíz)
    BASE = pathlib.Path(__file__).resolve().parents[1]

    paths = {
        "raw": BASE / "data" / "raw" / "geometric_figures",
        "interim_bw": BASE / "data" / "interim" / "bw",
        "interim_resize": BASE / "data" / "interim" / "resize",
        "interim_transparent": BASE / "data" / "interim" / "transparent",
        "processed_stats": BASE / "data" / "processed" / "stats",
    }

    print("Ejecutando pipeline completo...\n")
    ejecutar_pipeline(paths)
    print("\nPipeline FINALIZADO con éxito.")

if __name__ == "__main__":
    main()
