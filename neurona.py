import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Funciones principales
def predecir(X, pesos, sesgo):
    return np.dot(X, pesos) + sesgo

def error_cuadratico_medio(y_deseada, y_calculada):
    return np.mean((y_deseada - y_calculada)**2)

def entrenar(X, y, pesos, sesgo, tasa_aprendizaje, epocas):
    m = X.shape[0]
    
    historia_pesos = np.zeros((epocas, pesos.shape[0]))
    historia_sesgos = np.zeros(epocas)
    historia_costos = np.zeros(epocas)
    historia_predicciones = []

    for epoca in range(epocas):
        y_calculada = predecir(X, pesos, sesgo)
        
        error = y_calculada - y
        dw = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)
        
        pesos -= tasa_aprendizaje * dw
        sesgo -= tasa_aprendizaje * db
        
        costo = error_cuadratico_medio(y, y_calculada)
        historia_costos[epoca] = costo
        historia_pesos[epoca, :] = pesos
        historia_sesgos[epoca] = sesgo
        historia_predicciones.append(y_calculada)
    
    return pesos, sesgo, historia_pesos, historia_sesgos, historia_costos, historia_predicciones

# Cargar el dataset
def cargar_dataset(ruta_archivo):
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(f"El archivo '{ruta_archivo}' no se encontró.")
    data = pd.read_excel(ruta_archivo)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Configuración inicial de datos
ruta_dataset = 'datasets/221188.xlsx'
X, y_deseada = cargar_dataset(ruta_dataset)

# Valores iniciales aleatorios
np.random.seed(0)
pesos_iniciales = np.random.rand(X.shape[1])
sesgo_inicial = np.random.rand()

# Guardar y graficar resultados
def guardar_grafica_temp(y_deseada, y_predicha, epoca, carpeta):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_deseada, label='Y Deseada')
    plt.plot(y_predicha, label='Y Predicha')
    plt.title(f'Época {epoca + 1}')
    plt.legend()
    plt.tight_layout()
    archivo_imagen = f'{carpeta}/epoca_{epoca + 1}.png'
    plt.savefig(archivo_imagen)
    plt.close()
    return archivo_imagen

def crear_video(imagenes, archivo_video):
    frame = cv2.imread(imagenes[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(archivo_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for imagen in imagenes:
        video.write(cv2.imread(imagen))

    cv2.destroyAllWindows()
    video.release()

    # Eliminar imágenes temporales
    for imagen in imagenes:
        os.remove(imagen)

# Entrenar y actualizar
def entrenar_y_actualizar():
    global pesos_iniciales, sesgo_inicial
    
    pesos_iniciales = np.random.rand(X.shape[1])
    sesgo_inicial = np.random.rand()
    
    pesos = pesos_iniciales.copy()
    sesgo = sesgo_inicial

    tasa_aprendizaje = float(tasa_aprendizaje_entry.get())
    epocas = int(epocas_entry.get())
    
    pesos, sesgo, historia_pesos, historia_sesgos, historia_costos, historia_predicciones = entrenar(X, y_deseada, pesos, sesgo, tasa_aprendizaje, epocas)
    
    y_calculada = predecir(X, pesos, sesgo)
    costo_final = error_cuadratico_medio(y_deseada, y_calculada)
    
    # Actualizar tabla de pesos y sesgo
    for i in range(len(pesos)):
        tabla.set(f"final_{i}", 1, f"{pesos_iniciales[i]:.4f}")  
        tabla.set(f"final_{i}", 2, f"{pesos[i]:.4f}")
    
    tabla.set("final_sesgo", 1, f"{sesgo_inicial:.4f}")  
    tabla.set("final_sesgo", 2, f"{sesgo:.4f}")
    
    # Limpiar figuras de matplotlib
    plt.close('all')
    
    # Guardar gráficas temporalmente y crear video
    carpeta_temporal = 'temp_graficas'
    imagenes = []
    for epoca in range(epocas):
        archivo_imagen = guardar_grafica_temp(y_deseada, historia_predicciones[epoca], epoca, carpeta_temporal)
        imagenes.append(archivo_imagen)
    
    archivo_video = 'evolucion_entrenamiento.mp4'
    crear_video(imagenes, archivo_video)
    
    # Gráfico de evolución de pesos y sesgo
    plt.figure(figsize=(10, 6))
    for i in range(pesos.shape[0]):
        plt.plot(historia_pesos[:, i], label=f'Peso {i+1}')
    plt.plot(historia_sesgos, label='Sesgo', linestyle='--')
    plt.xlabel('Época')
    plt.ylabel('Valor')
    plt.title('Evolución de Pesos y Sesgo')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Gráfico de diferencias
    graficar_diferencias(y_deseada, historia_predicciones, epocas)
    
    # Gráfico de error
    graficar_error(historia_costos)

# Interfaz gráfica
ventana = tk.Tk()
ventana.title('Entrenamiento del Modelo de Neurona')

# Estilos
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TEntry', font=('Helvetica', 12))

# Marco principal
mainframe = ttk.Frame(ventana, padding="20 20 20 20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Etiquetas y campos de entrada
ttk.Label(mainframe, text="Tasa de Aprendizaje:").grid(column=0, row=0, padx=10, pady=5)
tasa_aprendizaje_entry = ttk.Entry(mainframe)
tasa_aprendizaje_entry.grid(column=1, row=0, padx=10, pady=5)
tasa_aprendizaje_entry.insert(tk.END, '0.1')

ttk.Label(mainframe, text="Épocas:").grid(column=0, row=1, padx=10, pady=5)
epocas_entry = ttk.Entry(mainframe)
epocas_entry.grid(column=1, row=1, padx=10, pady=5)
epocas_entry.insert(tk.END, '1000')

# Botón para iniciar entrenamiento
entrenar_button = ttk.Button(mainframe, text="Entrenar", command=entrenar_y_actualizar)
entrenar_button.grid(column=0, row=2, columnspan=2, pady=10)

# Tabla de pesos y sesgo
columns = ('Característica', 'Inicial', 'Final')
tabla = ttk.Treeview(mainframe, columns=columns, show='headings')

for col in columns:
    tabla.heading(col, text=col)
    tabla.column(col, anchor=tk.CENTER)

pesos_y_sesgo = [('w1', '0', '0'), ('w2', '0', '0'), ('w3', '0', '0'), ('w4', '0', '0'), ('w5', '0', '0'), ('w6', '0', '0'), ('Sesgo', '0', '0')]
for idw, (caracteristica, inicial, final) in enumerate(pesos_y_sesgo):
    item_id = f"final_{idw}" if caracteristica != 'Sesgo' else "final_sesgo"
    tabla.insert("", "end", iid=item_id, values=(caracteristica, inicial, final))

tabla.grid(column=0, row=3, columnspan=2, pady=10)

# Configurar la expansión de las celdas
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

ventana.mainloop()
