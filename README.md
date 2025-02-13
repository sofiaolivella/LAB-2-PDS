# LAB-2-PDS

## Descripción

## Desarrollo


![alt](tablaY1.jpg)
![alt](TablaY2.jpg)
![alt](ManoY1.jpg)
![alt](ManoY2.jpg)

![alt](h1.png)
![alt](x1.png)
![alt](y1.png)
![alt](y1Python.png)
![alt](h2.png)
![alt](x2.png)
![alt](y2Python.png)

Para el estudio de la correlación cruzada que se puede obtener entre dos variables, se usan dos señales m1(n*Ts) = cos(2*pi*100*n*Ts) y un m2(n*Ts) = sen(2*pi*100*n*Ts) definidas ambas entre 0 y 9 (incluyendo el 0 pero no el 9) con un valor Ts = 1.25 ms. Para ello primeramente se definen ambas funciones en el código y hacemos uso de la función incluida en “numpy” que es “correlate”, esta usa de parámetros las dos señales y nos entrega una señal discreta la cuál podemos graficar. A continuación, se presenta el código de dicho cálculo con su respectiva gráfica de correlación cruzada en función del desplazamiento. 

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

![alt](correlacion.png)

Teniendo en cuenta que la correlación nos mide el que tan similares son las señales en los desplazamientos, deberíamos evidenciar que la señal seno y coseno están desfasadas 90° eso quiere decir que cuando tomamos un valor de desplazamiento 0 su correlación es de 0 al estar desfasadas; a medida que aumentemos dicho valor la correlación empezará a aumentar hasta llegar a esa “igualdad” que es cuando ya nos desfasamos los 90° y podríamos afirmar que están en fase, de igual manera pasará si disminuimos el valor del desplazamiento solo que será negativo. Como se ve en la gráfica, esta relación se cumple y tiene una forma senoidal.

La tercera parte del laboratorio consiste en calcular valores estadísticos descriptivos de una señal de electroencefalografía en función del tiempo y de frecuencia, esto por medio de la transformada de Fourier.  

Antes de cualquier tipo de cálculo, debemos de adquirir y graficar la señal electroencefalográfica desde la base de datos de “Physionet”, para ello se descargaron desde la plataforma dos archivos con datos de EEG de un estudio, referenciado en el presente documento [1], con nombres de “s01_ex01_s02.hea” y “s01_ex01_s02.dat” (archivos también incluidos en el presente directorio.  Por medio del siguiente código y aprovechando la librería wfdb, específicamente la función “rdrecord” que nos permite acceder a los datos de los archivos previamente mencionados, podemos realizar la gráfica de la señal electroencefalográfica evidenciada a continuación. 

        EEG = "s01_ex01_s02"
        
        # Leer la señal desde el archivo
        lecturasignal = wfdb.rdrecord(EEG)
        signal = lecturasignal.p_signal[:,0]  
        fs = lecturasignal.fs  
        numero_datos = len(signal) 
        muestreo=int(5*fs)
        
        # Grafica la señal
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
        
![alt](EEG.png)

A dicha señal EEG representada en color violeta, se le realiza una caracterización de sus datos estadísticos descriptivos como la media, la desviación estándar, el coeficiente de variación y hasta la cantidad “n” de datos. Todos estos cálculos son realizados por medio de sus respectivas funciones incluidas en “numpy”, como se muestra en el código a continuación junto con sus respectivos resultados. 

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
        # Resultados
        Esta señal no es periódica
        Media de la señal con librerias: 0.2370
        Longitud del vector con librerias: 1000
        Desviación estándar con librerias: 3.4127
        Coeficiente de variación con librerias: 14.3998

![alt](HistogramaSeñalT.png)
![alt](EspectroNormalizado.png)
![alt](DensidadEspectral1.png)
![alt](HistogramaEnFrecuencia.png)
![alt](transformadas.png)

## Instrucciones

## Requerimientos

- Python 3.11
- Spyder 6.0
- Librerias como: wfdb, matplotlib, numpy, scipy.stats

## Referencias

[1] Abo Alzahab, N., Di Iorio, A., Apollonio, L., Alshalak, M., Gravina, A., Antognoli, L., Baldi, M., Scalise, L., & Alchalabi, B. (2021). Auditory evoked potential EEG-Biometric dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/ps31-fc50.

## AUTORES
- Juan Diego Clavijo Fuentes
  est.juan.dclavijjo@unimilitar.edu.co
- Sofia Olivella Moreno
  est.sofia.olivella@unimilitar.edu.co



