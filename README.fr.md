# Analformer : une nouvelle architecture de transformateur pour le décodage EEG et l'analyse neuroscientifique

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md"><strong>français</strong></a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md">中文</a>
</div>

Ce référentiel fournit la mise en œuvre officielle de l'article Scientific Reports :

**Une nouvelle architecture de transformateur pour le décodage EEG et l'analyse neuroscientifique**

Hong Gi Yeom, Woo Sung Choi et Kyung-min An, *Rapports scientifiques* (2026).

DOI : [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer est une architecture basée sur Transformer conçue pour prendre en charge à la fois le décodage d'interface cerveau-ordinateur (BCI) haute performance et l'analyse neuroscientifique interprétable. Le modèle utilise des noyaux d'ondelettes de Morlet fixes et non entraînables dans un module d'intégration de patchs analytiques afin que les représentations intermédiaires puissent être analysées avec des outils d'analyse EEG familiers tels que des cartes temps-fréquence, des topographies, une analyse temps-fréquence (FTF) de valeur F et une connectivité fonctionnelle basée sur l'attention.

## Aperçu

Analformer a été évalué sur des ensembles de données EEG publics couvrant quatre paramètres de décodage :

- **BCI Competition IV 2a** : imagerie motrice (MI) à quatre classes
- **OpenBMI MI** : imagerie motrice à deux classes
- **OpenBMI ERP** : décodage potentiel événementiel à deux classes
- **OpenBMI SSVEP** : décodage du potentiel évoqué visuellement en régime permanent à quatre classes

L'idée principale est de préserver une structure EEG interprétable lors de l'intégration. Les filtres à ondelettes Morlet fixes extraient les caractéristiques spatio-temporelles-fréquences, l'encodeur Transformer apprend les relations entre ces caractéristiques et le responsable de classification prédit la classe BCI cible. Les résultats de l'analyse sont produits à partir des représentations internes et des pondérations d'attention du modèle.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## Structure du référentiel

- `Analformer_BCI_Comp_4_2a.ipynb` : implémentation de l'imagerie motrice BCI Competition IV 2a.
- `Analformer_OpenBMI_MI.ipynb` : implémentation de l'imagerie moteur OpenBMI.
- `Analformer_OpenBMI_ERP.ipynb` : Implémentation de l'ERP OpenBMI.
- `Analformer_OpenBMI_SSVEP.ipynb` : implémentation d’OpenBMI SSVEP.
- `figures/Graphical_abstract.png` : Résumé graphique d'Analformer.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png` : exemples de résultats d'analyse tirés de l'article.
- `requirements.txt` : Dépendances du package Python utilisées par les notebooks.

## Exigences techniques

Python 3.9+ est recommandé. Une installation PyTorch compatible CUDA est recommandée pour la formation.

Installez d'abord PyTorch en fonction de votre environnement CUDA. Par exemple :

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Installez ensuite les dépendances restantes :

```bash
pip install -r requirements.txt
```

## Préparation des données

Les ensembles de données ne sont pas inclus dans ce référentiel. Veuillez télécharger les ensembles de données publics à partir de leurs sources originales :

- Concours BCI IV 2a : [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI : [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

Chaque ordinateur portable attend des fichiers `.mat` compatibles HDF5 avec les tableaux `data` et `label`. Les blocs-notes chargent les fichiers de sujet en utilisant le modèle de dénomination suivant :

- Fichiers de formation : `A1T.mat`, `A2T.mat`, ...
- Fichiers d'évaluation : `A1E.mat`, `A2E.mat`, ...

Définissez le dossier de l'ensemble de données dans le dictionnaire `params` à l'intérieur de `main()` :

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

Le code suppose des données EEG à 250 Hz. Les ordinateurs portables OpenBMI utilisent 62 canaux, tandis que l'ordinateur portable BCI Competition IV 2a utilise 22 canaux.

## Comment utiliser

1. Ouvrez le bloc-notes correspondant à l'ensemble de données/au paradigme que vous souhaitez exécuter.
2. Mettez à jour `"root"` dans le dictionnaire `params` vers votre dossier d'ensemble de données local.
3. Vérifiez les paramètres d'expérience clés tels que `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads` et `att_ch`.
4. Exécutez les cellules du bloc-notes dans l’ordre.
5. Utilisez `"anal": 1` pour générer des résultats d'analyse, notamment des cartes temps-fréquence, des topographies, des cartes de valeur F et des visualisations de connectivité basées sur l'attention.

Pour les notebooks OpenBMI, `Pretraining(params)` est exécuté avant les réglages spécifiques au sujet. Dans le bloc-notes BCI Competition IV 2a, le flux de travail actuel utilise un chemin de point de contrôle pré-entraîné lors du saut de pré-entraînement ; mettez à jour `pretrained_path` avant d'exécuter ce bloc-notes.

## Exemples de résultats d'analyse

Les figures suivantes montrent des exemples représentatifs de résultats d’analyse tirés du document.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## Citation

Si vous utilisez ce code, veuillez citer :

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## Remerciements

Ce travail a été soutenu par un fonds de recherche de l’Université Chosun.

Certaines parties de cette implémentation ont été développées en référence au référentiel [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), distribué sous la licence GPL-3.0. Nous remercions les auteurs d’avoir rendu leur mise en œuvre accessible au public.

## Licence

Le code source est distribué sous la licence `LICENSE`. Les chiffres papier sont reproduits ici uniquement pour expliquer la mise en œuvre officielle ; veuillez vous référer à la [page de l'article](https://www.nature.com/articles/s41598-026-56405-9) pour les détails de la publication et les informations sur les licences des figures.
