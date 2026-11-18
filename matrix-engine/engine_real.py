import os
import re
import urllib.request
import numpy as np
import pandas as pd

print("=== INICIANDO MOTOR TENSORIAL CON CORPUS REAL DE TAKAHASHI ===")

def obtener_corpus_takahashi():
    url_corpus = "https://www.voynich.nu/transcr/takahashi.txt"
    archivo_local = "takahashi_corpus_raw.txt"
    
    if not os.path.exists(archivo_local):
        print("[*] Descargando archivo maestro del corpus Takahashi...")
        try:
            urllib.request.urlretrieve(url_corpus, archivo_local)
            print("[+] Descarga completada con éxito.")
        except Exception as e:
            print(f"[-] Error al descargar el corpus: {e}")
            return None
    return archivo_local

def parsear_corpus_por_folios(filepath):
    if not filepath or not os.path.exists(filepath):
        print("[-] No se encontró el archivo físico del corpus para parsear.")
        return pd.DataFrame()

    folios_data = {}
    folio_actual = "desconocido"
    regex_folio = re.compile(r'<f([0-9]+[r|v])>')
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for linea in f:
            match_folio = regex_folio.search(linea)
            if match_folio:
                folio_actual = f"f{match_folio.group(1)}"
                if folio_actual not in folios_data:
                    folios_data[folio_actual] = []
                continue
            
            if linea.strip().startswith(('#', '*', '%', '=')):
                continue
                
            tokens_linea = re.findall(r'[a-zA-Z]+', linea)
            if tokens_linea and folio_actual != "desconocido":
                folios_data[folio_actual].extend(tokens_linea)

    df_frecuencias = pd.DataFrame()
    for folio in folios_data:
        if folios_data[folio]:
            s = pd.Series(folios_data[folio]).value_counts()
            df_frecuencias[folio] = s
            
    df_frecuencias = df_frecuencias.fillna(0).T
    return df_frecuencias

# Vocabulario maestro integrado (34 entidades: Grupo A + Grupo B)
grupo_a = [
    'sol', 'luna', 'stella', 'signum', 'zodiacus', 
    'aequinoctium', 'solstitium', 'ver', 'aestas', 'autumnus', 'hiems', 
    'plenilunium', 'interlunium', 'ascensio', 'descensio', 'augmentum', 'decrementum'
]
grupo_b = [
    'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 
    'libra', 'scorpio', 'sagittarius', 'capricornus', 'aquarius', 'pisces',
    'sol', 'luna', 'mercurius', 'venus', 'mars', 'iupiter', 'saturnus'
]
vocabulario_maestro = sorted(list(set(grupo_a + grupo_b)))

def ejecutar_motor_real():
    archivo = obtener_corpus_takahashi()
    T_df = parsear_corpus_por_folios(archivo)
    
    if T_df.empty:
        print("[-] La matriz T está vacía.")
        return

    print(f"\n[Dimensiones Matriz T Real]: {T_df.shape} (Folios x Tokens EVA)")
    
    folios_presentes = T_df.index.tolist()
    np.random.seed(42)
    L_raw = np.random.binomial(n=1, p=0.3, size=(len(vocabulario_maestro), len(folios_presentes)))
    L_df = pd.DataFrame(L_raw, index=vocabulario_maestro, columns=folios_presentes)
    
    V_values = np.dot(L_df.values, T_df.values)
    V = pd.DataFrame(V_values, index=vocabulario_maestro, columns=T_df.columns)
    
    print(f"\n=== MATRIZ V RESULTANTE (Dimensiones: {V.shape}) ===")
    print(V.head(10))
    
    total_celdas = V.size
    anuladas = (V == 0).sum().sum()
    print(f"\n--- AUDITORÍA DE ANULACIONES GLOBALES ---")
    print(f"Celdas con anulación absoluta (V_ij = 0): {anuladas} ({anuladas/total_celdas*100:.2f}%)")

if __name__ == "__main__":
    ejecutar_motor_real()
