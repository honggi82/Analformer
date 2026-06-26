# Analformer: EEG デコードと神経科学分析のための新しいトランスフォーマー アーキテクチャ

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md"><strong>日本語</strong></a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md">中文</a>
</div>

このリポジトリは、Scientific Reports の記事の公式実装を提供します。

**EEG デコードと神経科学分析のための新しいトランスフォーマー アーキテクチャ**

ホン・ギヨム、チェ・ウソン、アン・ギョンミン、*Scientific Reports* (2026)。

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer は、高性能ブレイン コンピューター インターフェイス (BCI) デコードと解釈可能な神経科学分析の両方をサポートするように設計された Transformer ベースのアーキテクチャです。このモデルは、分析パッチ埋め込みモジュールで固定のトレーニング不可能な Morlet ウェーブレット カーネルを使用するため、時間-周波数マップ、トポグラフィ、F 値時間-周波数 (FTF) 分析、注意ベースの機能接続などの使い慣れた EEG 分析ツールで中間表現を分析できます。

## 概要

Analformer は、4 つのデコード設定をカバーする公開 EEG データセットで評価されました。

- **BCI Competition IV 2a**: 4 クラスのモーターイメージ (MI)
- **OpenBMI MI**: 2 クラスの運動イメージ
- **OpenBMI ERP**: 2 クラスのイベント関連の潜在的なデコード
- **OpenBMI SSVEP**: 4 クラスの定常状態の視覚的に誘発される潜在的なデコーディング

中心的なアイデアは、埋め込み中に解釈可能な EEG 構造を保存することです。固定モーレット ウェーブレット フィルターは時空間周波数特徴を抽出し、Transformer エンコーダーはこれらの特徴間の関係を学習し、分類ヘッドはターゲット BCI クラスを予測します。分析出力は、モデルの内部表現とアテンションの重みから生成されます。

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## リポジトリ構造

- `Analformer_BCI_Comp_4_2a.ipynb`: BCI Competition IV 2a モーター イメージの実装。
- `Analformer_OpenBMI_MI.ipynb`: OpenBMI Motor Imagery の実装。
- `Analformer_OpenBMI_ERP.ipynb`: OpenBMI ERP の実装。
- `Analformer_OpenBMI_SSVEP.ipynb`: OpenBMI SSVEP 実装。
- `figures/Graphical_abstract.png`: Analformer のグラフィック概要。
- `figures/Fig4.png`、`figures/Fig5.png`、`figures/Fig7.png`: 論文の解析結果の例。
- `requirements.txt`: ノートブックで使用される Python パッケージの依存関係。

## 技術的要件

Python 3.9 以降を推奨します。トレーニングには CUDA 対応の PyTorch インストールをお勧めします。

CUDA 環境に応じて、最初に PyTorch をインストールします。たとえば:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

次に、残りの依存関係をインストールします。

```bash
pip install -r requirements.txt
```

## データの準備

データセットはこのリポジトリには含まれていません。公開データセットは元のソースからダウンロードしてください。

- BCI コンペティション IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

各ノートブックでは、`data` および `label` 配列を含む HDF5 互換の `.mat` ファイルが必要です。ノートブックは、次の命名パターンを使用してサブジェクト ファイルを読み込みます。

- トレーニング ファイル: `A1T.mat`、`A2T.mat`、...
- 評価ファイル: `A1E.mat`、`A2E.mat`、...

`main()` 内の `params` ディクショナリにデータセット フォルダーを設定します。

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

コードは 250 Hz の EEG データを想定しています。 OpenBMI ノートブックは 62 チャネルを使用しますが、BCI Competition IV 2a ノートブックは 22 チャネルを使用します。

## 使用方法

1. 実行するデータセット/パラダイムのノートブックを開きます。
2. `params` ディクショナリの `"root"` をローカル データセット フォルダーに更新します。
3. `n_ch`、`n_classes`、`time`、`baseline_sec`、`pretrain_epochs`、`finetuning_epochs`、`depth`、`num_heads`、`att_ch` などの主要な実験パラメーターを確認します。
4. ノートブックのセルを順番に実行します。
5. `"anal": 1` を使用して、時間-周波数マップ、地形、F 値マップ、アテンションベースの接続視覚化などの解析出力を生成します。

OpenBMI ノートブックの場合、`Pretraining(params)` は主題固有の微調整の前に実行されます。 BCI Competition IV 2a ノートブックでは、現在のワークフローは、事前トレーニングをスキップするときに、事前トレーニングされたチェックポイント パスを使用します。そのノートブックを実行する前に、`pretrained_path` を更新してください。

## 解析結果例

以下の図は論文の代表的な解析結果例を示しています。

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## 引用

このコードを使用する場合は、以下を引用してください。

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## 謝辞

この研究は朝鮮大学からの研究資金によって支援されました。

この実装の一部は、GPL-3.0 ライセンスの下で配布されている [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) リポジトリを参照して開発されました。実装を公開してくださった作者に感謝します。

## ライセンス

ソースコードは`LICENSE`のライセンスに基づいて配布されています。紙上の図は、正式な実装を説明するためにのみここに転載されています。出版物の詳細とフィギュアのライセンス情報については、[記事ページ](https://www.nature.com/articles/s41598-026-56405-9)を参照してください。
