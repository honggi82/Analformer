# Analformer: una novedosa arquitectura transformadora para decodificación de EEG y análisis neurocientífico

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md"><strong>Español</strong></a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md">中文</a>
</div>

Este repositorio proporciona la implementación oficial del artículo de Scientific Reports:

**Una nueva arquitectura de transformador para decodificación de EEG y análisis neurocientífico**

Hong Gi Yeom, Woo Sung Choi y Kyung-min An, *Scientific Reports* (2026).

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer es una arquitectura basada en Transformer diseñada para admitir tanto la decodificación de la interfaz cerebro-computadora (BCI) de alto rendimiento como el análisis neurocientífico interpretable. El modelo utiliza núcleos de ondas Morlet fijos y no entrenables en un módulo de incrustación de parches analíticos para que las representaciones intermedias se puedan analizar con herramientas de análisis de EEG familiares, como mapas de tiempo-frecuencia, topografías, análisis de tiempo-frecuencia (FTF) de valor F y conectividad funcional basada en la atención.

## Descripción general

Analformer se evaluó en conjuntos de datos públicos de EEG que cubren cuatro configuraciones de decodificación:

- **Competencia BCI IV 2a**: imágenes motoras (MI) de cuatro clases
- **OpenBMI MI**: imágenes motoras de dos clases
- **OpenBMI ERP**: decodificación potencial relacionada con eventos de dos clases
- **OpenBMI SSVEP**: decodificación de potencial evocado visualmente en estado estacionario de cuatro clases

La idea central es preservar la estructura EEG interpretable durante la incorporación. Los filtros de ondas Morlet fijos extraen características de frecuencia espacio-temporal, el codificador Transformer aprende las relaciones entre estas características y el cabezal de clasificación predice la clase BCI objetivo. Los resultados del análisis se producen a partir de las representaciones internas y los pesos de atención del modelo.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## Estructura del repositorio

- `Analformer_BCI_Comp_4_2a.ipynb`: Implementación de Imágenes Motorizadas IV Competencia BCI 2a.
- `Analformer_OpenBMI_MI.ipynb`: Implementación de OpenBMI Motor Imagery.
- `Analformer_OpenBMI_ERP.ipynb`: Implementación del ERP OpenBMI.
- `Analformer_OpenBMI_SSVEP.ipynb`: Implementación de OpenBMI SSVEP.
- `figures/Graphical_abstract.png`: Resumen gráfico de Analformer.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png`: ejemplos de resultados de análisis del artículo.
- `requirements.txt`: Dependencias del paquete Python utilizadas por los cuadernos.

## Requisitos técnicos

Se recomienda Python 3.9+. Se recomienda una instalación de PyTorch habilitada para CUDA para la capacitación.

Instale PyTorch primero de acuerdo con su entorno CUDA. Por ejemplo:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Luego instale las dependencias restantes:

```bash
pip install -r requirements.txt
```

## Preparación de datos

Los conjuntos de datos no están incluidos en este repositorio. Descargue los conjuntos de datos públicos de sus fuentes originales:

- Concurso BCI IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- IMC abierto: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

Cada portátil espera archivos `.mat` compatibles con HDF5 con matrices `data` y `label`. Los cuadernos cargan archivos de temas utilizando el siguiente patrón de nomenclatura:

- Archivos de formación: `A1T.mat`, `A2T.mat`, ...
- Archivos de evaluación: `A1E.mat`, `A2E.mat`, ...

Configure la carpeta del conjunto de datos en el diccionario `params` dentro de `main()`:

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

El código asume datos de EEG de 250 Hz. Los portátiles OpenBMI utilizan 62 canales, mientras que el portátil BCI Competition IV 2a utiliza 22 canales.

## Cómo utilizar

1. Abra el cuaderno del conjunto de datos/paradigma que desea ejecutar.
2. Actualice `"root"` en el diccionario `params` en su carpeta de conjunto de datos local.
3. Verifique los parámetros clave del experimento, como `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads` y `att_ch`.
4. Ejecute las celdas del cuaderno en orden.
5. Utilice `"anal": 1` para generar resultados de análisis, incluidos mapas de tiempo-frecuencia, topografías, mapas de valores F y visualizaciones de conectividad basadas en la atención.

Para los cuadernos OpenBMI, `Pretraining(params)` se ejecuta antes del ajuste específico del tema. En el cuaderno BCI Competition IV 2a, el flujo de trabajo actual utiliza una ruta de punto de control previamente entrenada al omitir el entrenamiento previo; actualice `pretrained_path` antes de ejecutar esa computadora portátil.

## Ejemplos de resultados de análisis

Las siguientes figuras muestran ejemplos representativos de resultados de análisis del artículo.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## Citación

Si utiliza este código, cite:

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## Agradecimientos

Este trabajo fue apoyado por un fondo de investigación de la Universidad Chosun.

Partes de esta implementación se desarrollaron con referencia al repositorio [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), que se distribuye bajo la licencia GPL-3.0. Agradecemos a los autores por hacer pública su implementación.

## Licencia

El código fuente se distribuye bajo la licencia `LICENSE`. Las cifras en papel se reproducen aquí sólo para explicar la implementación oficial; consulte la [página del artículo](https://www.nature.com/articles/s41598-026-56405-9) para obtener detalles de la publicación e información sobre la licencia de figuras.
