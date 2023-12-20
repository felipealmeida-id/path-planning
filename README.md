# Generador de Trayectorias de UAV con GAN
Este repositorio contiene un modelo de Generative Adversarial Network (GAN) diseñado para generar trayectorias realistas para vehículos aéreos no tripulados (UAV). La GAN se entrena utilizando datos de trayectorias de UAV existentes para aprender patrones y características que luego se pueden utilizar para generar nuevas trayectorias.

## Descripción del Proyecto
El objetivo principal de este proyecto es proporcionar una herramienta que permita la generación automática de trayectorias de vuelo para UAV. Esto puede ser útil en escenarios donde se requiere una planificación de vuelo eficiente y adaptativa, como en misiones de vigilancia, exploración o mapeo.

## Estructura del Repositorio
downscaler/: Lo correspondiente al rescalador de rutas

envs/: Los distintos ambientes para entrenar a la red

gan_perceptron:/ Lo que compete a la red neuronal

## Requisitos del Sistema
Asegúrate de tener instaladas las siguientes bibliotecas y dependencias antes de ejecutar el código:
```bash
conda create --name <env> --file requirements.txt
```
## Uso
Para llevar a cabo entrenamiento de la red se utiliza el comando
```bash
python main.py newCartesian gan
```
Esto resultara en estados de la red, y salidas del generador guardadas en las carpetas 
1. `output/newCartesian/gan/generator` 
2. `output/newCartesian/gan/discriminator`
3. `output/newCartesian/gan/generated_imgs`

Para visualizar una trayectoria lo recomndado es copiar del archivo encontrado en la carpeta `output/newCartesian/gan/generated_imgs` la trayectoria, y luego copiarla en el archivo `cartesianDraw.py`. Con este comando se dibujara la trayectoria:
```bash
python cartesianDraw.py
``` 
