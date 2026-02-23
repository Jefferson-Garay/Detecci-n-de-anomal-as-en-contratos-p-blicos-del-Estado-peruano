import pandas as pd
import os, time, gc, pickle, numpy as np, re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import pytesseract
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


# ===========================================================
# FUNCIONES
# ===========================================================

def descargar_archivo(url, carpeta="descargas"):
    print(f"Descargando desde {url}...")
    os.makedirs(carpeta, exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(carpeta),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True
    })

    d = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                         options=options)

    antes = set(os.listdir(carpeta))
    d.get(url)

    archivo = None
    for _ in range(120):
        time.sleep(1)
        nuevos = set(os.listdir(carpeta)) - antes
        nuevos = [f for f in nuevos if not f.endswith((".crdownload", ".tmp"))]
        if nuevos:
            archivo = nuevos[0]
            break

    d.quit()

    if archivo:
        ruta = os.path.join(carpeta, archivo)
        print("PDF descargado en:", ruta)
        return ruta

    print("Descarga fallida")
    return None


def extract_text_from_pdf_ocr(pdf_path, dpi=300, lang="spa"):
    try:
        temp_folder = "temp_img_ocr"
        os.makedirs(temp_folder, exist_ok=True)

        info = pdfinfo_from_path(pdf_path)
        page_count = info['Pages']

        texto = ""

        for i in range(1, page_count + 1):
            pages = convert_from_path(pdf_path, dpi=dpi,
                                      first_page=i, last_page=i)
            img = pages[0]

            img_path = os.path.join(temp_folder, f"page_{i}.png")
            img.save(img_path, "PNG")

            texto += pytesseract.image_to_string(Image.open(img_path),
                                                 lang=lang)

            os.remove(img_path)
            del img
            gc.collect()

        os.removedirs(temp_folder)
        return texto

    except Exception as e:
        return f"Error OCR: {e}"


def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü0-9 ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    stop_es = set(stopwords.words("spanish"))
    return " ".join([p for p in texto.split() if p not in stop_es and len(p) > 2])


def obtener_cluster(texto_pdf, ruta_modelo_cluster="modelo_cluster.pkl"):
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb = modelo.encode([texto_pdf])         # <--- Embedding del OCR (solo para cluster)

    with open(ruta_modelo_cluster, "rb") as f:
        kmeans = pickle.load(f)

    cluster = kmeans.predict(emb)[0]
    return cluster


def generar_embedding_descripcion(texto):
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb = modelo.encode([texto])[0]           # vector 384
    return emb


# ===========================================================
# MAIN PIPELINE
# ===========================================================

# ===========================================================
# MAIN PIPELINE (CORREGIDO)
# ===========================================================
if __name__ == "__main__":

    # ================================
    # 0) Cargar Excel completo
    # ================================
    data = pd.read_excel("test.xlsx")

    # Copia para ir agregando columnas nuevas
    resultados = []

    # Cargar modelo final
    with open("modelo_anomalia_regresion.pkl", "rb") as f:
        modelo_final = pickle.load(f)

    # Iterar por todas las filas del Excel
    for idx, fila in data.iterrows():

        print(f"\n==============================")
        print(f"Procesando fila {idx+1}/{len(data)}")
        print("==============================")

        # 1) Descargar PDF
        ruta_pdf = descargar_archivo(fila["urlcontrato"])
        if not ruta_pdf:
            print("❌ No se pudo descargar el PDF, se marca como NaN")
            pred = np.nan
            resultados.append(pred)
            continue

        # 2) OCR
        texto_pdf = extract_text_from_pdf_ocr(ruta_pdf)

        # 3) Cluster desde OCR
        cluster = obtener_cluster(texto_pdf)

        # 4) Embedding descripción_proceso
        emb_desc = generar_embedding_descripcion(fila["descripcion_proceso"])
        emb_df = pd.DataFrame([emb_desc], columns=[f"emb_{i}" for i in range(len(emb_desc))])

        # 5) Construir fila con estructura final
        fila_proc = pd.DataFrame([fila]).drop(columns=["descripcion_proceso", "urlcontrato"])

        fila_proc["cluster"] = cluster

        # Concatenar embeddings
        fila_final = pd.concat([fila_proc.reset_index(drop=True), emb_df], axis=1)

        # 6) Reemplazar NaNs
        fila_final = fila_final.fillna(0)

        # 7) Reordenar columnas EXACTAMENTE como entrenamiento
        columnas_finales = (
            ["monto_contratado_total",
             "monto_adicional",
             "monto_reduccion",
             "monto_prorroga",
             "tieneresolucion",
             "duracion_contrato_dias",
             "cluster"]
            +
            [f"emb_{i}" for i in range(384)]
        )

        fila_final = fila_final.reindex(columns=columnas_finales, fill_value=0)

        # 8) Convertir a numpy
        X_input = fila_final.values

        # 9) Predicción
        pred = modelo_final.predict(X_input)[0]
        resultados.append(pred)

        print(f"Predicción: {pred}")

    # Guardar resultados en nuevo Excel
    data["prediccion_anomalia"] = resultados
    data.to_excel("predicciones_output.xlsx", index=False)

    print("\n=======================================")
    print("Predicciones completadas y guardadas en:")
    print(" → predicciones_output.xlsx")
    print("=======================================")



