import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Gráfica de las señales de Juan Diego

h1=np.array([5,6,0,0,7,1,5]);
x1=np.array([1,0,0,7,8,1,2,2,0,2]);
y1=np.array([5,6,0,35,89,54,21,71,75,60,67,21,12,24,2,10]);

t1 = np.arange(len(h1)) 
plt.figure(figsize=(8, 4))
plt.stem(t1, h1)
plt.xlabel('n')
plt.ylabel('h1(n)')
plt.title('Gráfica de señal h1(n)')
plt.grid()
plt.show()

t2 = np.arange(len(x1)) 
plt.figure(figsize=(8, 4))
plt.stem(t2, x1)
plt.xlabel('n')
plt.ylabel('x1(n)')
plt.title('Gráfica de señal x1(n)')
plt.grid()
plt.show()

# Gráfico convolución a mano

t3 = np.arange(len(y1)) 
plt.figure(figsize=(8, 4))
plt.stem(t3, y1)
plt.xlabel('n')
plt.ylabel('y1(n)')
plt.title('Gráfica del resultado a mano de la convolución y1(n)')
plt.grid()
plt.show()

# Gráfico convolución por python

y11 = np.convolve(x1, h1, mode='full');

t7 = np.arange(len(y11)) 
plt.figure(figsize=(8, 4))
plt.stem(t7, y11)
plt.xlabel('n')
plt.ylabel('y1(n)')
plt.title('Gráfica de la convolución y1(n) por Python')
plt.grid()
plt.show()

# Gráfica de las señales se Sofia
h2=np.array([5,6,0,0,6,8,4]);
x2=np.array([1,0,3,1,6,4,3,9,2,9]);
y2=np.array([5,6,15,23,42,64,61,93,120,133,128,94,96,106,80,36]);

t4 = np.arange(len(h2)) 
plt.figure(figsize=(8, 4))
plt.stem(t4, h2)
plt.xlabel('n')
plt.ylabel('h2(n)')
plt.title('Gráfica de señal h2(n)')
plt.grid()
plt.show()

t5 = np.arange(len(x2)) 
plt.figure(figsize=(8, 4))
plt.stem(t5, x2)
plt.xlabel('n')
plt.ylabel('x2(n)')
plt.title('Gráfica de señal x2(n)')
plt.grid()
plt.show()

# Gráfico convolución a mano

t6 = np.arange(len(y2)) 
plt.figure(figsize=(8, 4))
plt.stem(t6, y2)
plt.xlabel('n')
plt.ylabel('y2(n)')
plt.title('Gráfica del resultado de la convolución a mano de y2(n)')
plt.grid()
plt.show()

# Gráfico convolución con python

y22 = np.convolve(x2, h2, mode='full');

t8 = np.arange(len(y22)) 
plt.figure(figsize=(8, 4))
plt.stem(t8, y22)
plt.xlabel('n')
plt.ylabel('y2(n)')
plt.title('Gráfica de la convolución y2(n) por Python')
plt.grid()
plt.show()

# CORRELACIONES

Ts = 1.25e-3;
n = np.arange(9);

m1 = np.cos(2*3.1416*100*n*Ts);
m2 = np.sin(2*3.1416*100*n*Ts);

cor = np.correlate(m1, m2, mode='full')
print("Correlación entre las señales m1 y m2", cor)

# Graficar Correlacion entre ambas señales
t9 = np.arange(-len(n) + 1, len(n))
plt.figure(figsize=(8, 4))
plt.stem(t9, cor)
plt.xlabel('Desplazamiento')
plt.ylabel('Correlación')
plt.title('Correlación cruzada entre m1(n) y m2(n)')
plt.grid()
plt.show()


EEG = "s01_ex01_s02"

# Leer la señal desde el archivo
lecturasignal = wfdb.rdrecord(EEG)
signal = lecturasignal.p_signal[:,0]  
fs = lecturasignal.fs  
numero_datos = len(signal) 
muestreo=int(5*fs)

# Grafica la señal muscular del gastrocnemio
time = [i / fs for i in range(numero_datos)]  
signal = signal[:muestreo]
time = time[:muestreo]
plt.figure(figsize=(12,4))
plt.plot(time, signal, color="violet")

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mv)")
plt.title("Señal Biomédica EEG bases de datos physionet")
plt.grid()
plt.show()


# Histograma solo

plt.figure(figsize=(8, 4))
plt.hist(signal, bins=50, color='orange', alpha=0.7, edgecolor='black', density=True)
plt.xlabel("Amplitud de la señal")
plt.ylabel("Frecuencia normalizada")
plt.title("Histograma de la señal")
plt.grid()

# Función de probabilidad
kde = gaussian_kde(signal)
x_vals = np.linspace(min(signal), max(signal), 1000)
pdf_vals = kde(x_vals)
plt.plot(x_vals, pdf_vals, color='brown', label="")
plt.show()


# calculos con funciones de python
media_librerias = np.mean(signal)
longitud_vector_librerias = len(signal)
desviacion_librerias = np.std(signal)
coeficiente_variacion_librerias = (desviacion_librerias / media_librerias) if media_librerias != 0 else np.nan

print("Esta señal no es periódica")
print(f"Media de la señal con librerias: {media_librerias:.4f}")
print(f"Longitud del vector con librerias: {longitud_vector_librerias}")
print(f"Desviación estándar con librerias: {desviacion_librerias:.4f}")
print(f"Coeficiente de variación con librerias: {coeficiente_variacion_librerias:.4f}")

# Transformada de Fourier

t = np.linspace(0, 1, fs, endpoint=False) 
N = len(t)

frequencies = np.fft.fftfreq(N, 1/fs)
spectrum = np.fft.fft(signal) / N
magnitud = 2 * np.abs(spectrum[:N//2]) 


plt.figure(figsize=(12,4))
plt.plot(frequencies[:N//2], magnitud, 'orange')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Espectro de la señal normalizado')
plt.grid()
plt.show()

psd = (magnitud ** 2) / N

plt.figure(figsize=(12,4))
plt.plot(frequencies[:N//2], psd, 'violet')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral')
plt.title('Espectro de la señal de Densidad Espectral')
plt.grid()
plt.show()


mediana_librerias = np.median(magnitud)
media_librerias = np.mean(magnitud)
desviacion_librerias = np.std(magnitud)

print(f"Media de la señal con respecto a la frecuencia es: {media_librerias:.4f}")
print(f"Desviación estándar con respecto a la frecuencia: {desviacion_librerias:.4f}")
print(f"La mediana en cuanto a la frecuencia : {mediana_librerias:.4f}")

plt.figure(figsize=(8, 4))
plt.hist(magnitud, bins=50, color='orange', alpha=0.7, edgecolor='black', density=True)
plt.xlabel("Magnitud")
plt.ylabel("Frecuencia (Hz)")
plt.title("Histograma de la Frecuencia")
plt.grid()
plt.show()


plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(signal, label="Señal electroencefalográfica", color="red")
plt.title("Señal electroencefalográfica")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(magnitud[:muestreo//2], label="Transformada de Fourier al EEG", color="blue")
plt.title("Transformada de Fourier al EEG")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(psd[:muestreo//2], label="Densidad espectral", color="black")
plt.title("Densidad espectral")
plt.legend()

plt.tight_layout()
plt.show()