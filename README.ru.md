# Analformer: новая архитектура преобразователя для декодирования ЭЭГ и нейробиологического анализа.

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md"><strong>Русский</strong></a> |
  <a href="README.zh.md">中文</a>
</div>

Этот репозиторий предоставляет официальную реализацию статьи Scientific Reports:

**Новая архитектура преобразователя для декодирования ЭЭГ и нейробиологического анализа**

Хон Ги Ём, У Сон Чхве и Кён Мин Ан, *Научные отчеты* (2026).

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer — это архитектура на основе Transformer, предназначенная для поддержки как высокопроизводительного декодирования интерфейса мозг-компьютер (BCI), так и интерпретируемого нейробиологического анализа. Модель использует фиксированные, необучаемые ядра вейвлетов Морле в модуле Analytical Patch Embedding, так что промежуточные представления можно анализировать с помощью знакомых инструментов анализа ЭЭГ, таких как карты частоты времени, топографии, анализ частоты времени F-значения (FTF) и функциональная связь на основе внимания.

## Обзор

Analformer оценивался на общедоступных наборах данных ЭЭГ, охватывающих четыре параметра декодирования:

- **Соревнование BCI IV 2a**: четыре класса двигательного воображения (MI)
- **OpenBMI MI**: два класса двигательных образов
- **OpenBMI ERP**: двухклассное потенциальное декодирование, связанное с событиями
- **OpenBMI SSVEP**: декодирование стационарного визуально вызванного потенциала с четырьмя классами

Основная идея состоит в том, чтобы сохранить интерпретируемую структуру ЭЭГ во время встраивания. Фиксированные вейвлет-фильтры Морле извлекают пространственно-временные частотные характеристики, кодер Transformer изучает взаимосвязи между этими функциями, а головка классификации прогнозирует целевой класс BCI. Результаты анализа получаются на основе внутренних представлений модели и весов внимания.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## Структура репозитория

- `Analformer_BCI_Comp_4_2a.ipynb`: реализация BCI Competition IV 2a Motor Imagery.
- `Analformer_OpenBMI_MI.ipynb`: реализация OpenBMI Motor Imagery.
- `Analformer_OpenBMI_ERP.ipynb`: реализация ERP OpenBMI.
- `Analformer_OpenBMI_SSVEP.ipynb`: реализация OpenBMI SSVEP.
- `figures/Graphical_abstract.png`: Графическое описание Analformer.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png`: примеры результатов анализа из статьи.
- `requirements.txt`: зависимости пакетов Python, используемые блокнотами.

## Технические требования

Рекомендуется Python 3.9+. Для обучения рекомендуется установка PyTorch с поддержкой CUDA.

Сначала установите PyTorch в соответствии с вашей средой CUDA. Например:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Затем установите оставшиеся зависимости:

```bash
pip install -r requirements.txt
```

## Подготовка данных

Наборы данных не включены в этот репозиторий. Пожалуйста, загрузите общедоступные наборы данных из первоначальных источников:

- Соревнования BCI IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

Каждый ноутбук ожидает HDF5-совместимые файлы `.mat` с массивами `data` и `label`. Блокноты загружают тематические файлы, используя следующий шаблон именования:

- Учебные файлы: `A1T.mat`, `A2T.mat`,...
- Оценочные файлы: `A1E.mat`, `A2E.mat`, ...

Установите папку набора данных в словаре `params` внутри `main()`:

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

Код предполагает данные ЭЭГ с частотой 250 Гц. Ноутбуки OpenBMI используют 62 канала, а ноутбук BCI Competition IV 2a — 22 канала.

## Как использовать

1. Откройте блокнот для набора данных/парадигмы, которую вы хотите запустить.
2. Обновите `"root"` в словаре `params` в локальной папке набора данных.
3. Проверьте ключевые параметры эксперимента, такие как `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads` и `att_ch`.
4. Проведите ячейки блокнота по порядку.
5. Используйте `"anal": 1` для создания результатов анализа, включая карты частоты времени, топографии, карты F-значений и визуализации соединений на основе внимания.

Для ноутбуков OpenBMI `Pretraining(params)` выполняется перед тонкой настройкой конкретного объекта. В записной книжке BCI Competition IV 2a текущий рабочий процесс использует предварительно обученный путь контрольной точки при пропуске предварительного обучения; обновите `pretrained_path` перед запуском этого ноутбука.

## Примеры результатов анализа

На следующих рисунках показаны репрезентативные примеры результатов анализа из статьи.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## Цитирование

Если вы используете этот код, укажите:

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## Благодарности

Эта работа была поддержана исследовательским фондом Университета Чосон.

Части этой реализации были разработаны со ссылкой на репозиторий [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), который распространяется по лицензии GPL-3.0. Мы благодарим авторов за то, что они сделали свою реализацию общедоступной.

## Лицензия

Исходный код распространяется по лицензии `LICENSE`. Бумажные цифры воспроизводятся здесь только для объяснения официальной реализации; пожалуйста, обратитесь к [странице статьи](https://www.nature.com/articles/s41598-026-56405-9) для получения подробной информации о публикации и информации о лицензировании рисунков.
