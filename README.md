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

Teniendo en cuenta que la correlación nos mide el que tan similares son las señales en los desplazamientos, deberíamos evidenciar que la señal seno y coseno están desfasadas 90° eso quiere decir que cuando tomamos un valor de desplazamiento 0 su correlación es de 0 al estar desfasadas; a medida que aumentemos dicho valor la correlación empezará a aumentar hasta llegar a esa “igualdad” que es cuando ya nos desfasamos los 90° y podríamos afirmar que están en fase, de igual manera pasará si disminuimos el valor del desplazamiento solo que será negativo. 

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
![alt](EEG.png)
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

## AUTORES
- Juan Diego Clavijo Fuentes
  est.juan.dclavijjo@unimilitar.edu.co
- Sofia Olivella Moreno
  est.sofia.olivella@unimilitar.edu.co



