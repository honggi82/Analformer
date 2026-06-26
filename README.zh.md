# Analformer：一种用于脑电图解码和神经科学分析的新型变压器架构

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md"><strong>中文</strong></a>
</div>

该存储库提供了科学报告文章的官方实现：

**用于脑电图解码和神经科学分析的新颖变压器架构**

Hong Gi Yeom、Woo Sung Choi 和 Kyung-min An，*科学报告* (2026)。

DOI：[10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer 是一种基于 Transformer 的架构，旨在支持高性能脑机接口 (BCI) 解码和可解释的神经科学分析。该模型在分析补丁嵌入模块中使用固定的、不可训练的 Morlet 小波核，以便可以使用熟悉的 EEG 分析工具（例如时频图、地形图、F 值时频 (FTF) 分析和基于注意力的功能连接）来分析中间表示。

## 概述

Analformer 在公共脑电图数据集上进行了评估，涵盖四种解码设置：

- **BCI 竞赛 IV 2a**：四级运动想象 (MI)
- **OpenBMI MI**：两类运动想象
- **OpenBMI ERP**：两类事件相关电位解码
- **OpenBMI SSVEP**：四级稳态视觉诱发电位解码

核心思想是在嵌入过程中保留可解释的脑电图结构。固定 Morlet 小波滤波器提取时空频率特征，Transformer 编码器学习这些特征之间的关系，分类头预测目标 BCI 类别。分析输出是根据模型的内部表示和注意力权重产生的。

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## 存储库结构

- `Analformer_BCI_Comp_4_2a.ipynb`：BCI 竞赛 IV 2a 运动意象实施。
- `Analformer_OpenBMI_MI.ipynb`：OpenBMI 运动图像实现。
- `Analformer_OpenBMI_ERP.ipynb`：OpenBMI ERP 实施。
- `Analformer_OpenBMI_SSVEP.ipynb`：OpenBMI SSVEP 实现。
- `figures/Graphical_abstract.png`：Analformer 的图形摘要。
- `figures/Fig4.png`、`figures/Fig5.png`、`figures/Fig7.png`：论文中的分析结果示例。
- `requirements.txt`：笔记本使用的 Python 包依赖项。

## 技术要求

推荐使用Python 3.9+。建议安装支持 CUDA 的 PyTorch 进行培训。

首先根据自己的CUDA环境安装PyTorch。例如：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后安装剩余的依赖项：

```bash
pip install -r requirements.txt
```

## 数据准备

数据集不包含在此存储库中。请从原始来源下载公共数据集：

- BCI 竞赛 IV 2a：[http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

每个笔记本电脑都需要具有 `data` 和 `label` 阵列的 HDF5 兼容 `.mat` 文件。笔记本使用以下命名模式加载主题文件：

- 训练文件：`A1T.mat`、`A2T.mat`，...
- 评估文件：`A1E.mat`、`A2E.mat`，...

在`main()`内部的`params`字典中设置数据集文件夹：

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

该代码假设 EEG 数据为 250 Hz。 OpenBMI 笔记本电脑使用 62 个通道，而 BCI Comparison IV 2a 笔记本电脑使用 22 个通道。

## 如何使用

1. 打开您要运行的数据集/范式的笔记本。
2. 将 `params` 字典中的 `"root"` 更新到本地数据集文件夹。
3. 检查`n_ch`、`n_classes`、`time`、`baseline_sec`、`pretrain_epochs`、`finetuning_epochs`、`depth`、`num_heads`、`att_ch`等关键实验参数。
4. 按顺序运行笔记本单元。
5. 使用 `"anal": 1` 生成分析输出，包括时频图、地形图、F 值图和基于注意力的连接可视化。

对于 OpenBMI 笔记本，`Pretraining(params)` 在特定于主题的微调之前执行。在 BCI Comparison IV 2a 笔记本中，当前工作流程在跳过预训练时使用预训练的检查点路径；在运行该笔记本之前更新 `pretrained_path`。

## 分析结果示例

下图显示了论文中代表性的分析结果示例。

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## 引文

如果您使用此代码，请引用：

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## 致谢

这项工作得到了朝鲜大学研究基金的支持。

该实现的部分内容是参考 [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) 存储库开发的，该存储库在 GPL-3.0 许可证下分发。我们感谢作者公开他们的实现。

## 执照

源代码根据 `LICENSE` 许可证分发。这里转载纸质数据只是为了解释官方的实现；请参阅[文章页面](https://www.nature.com/articles/s41598-026-56405-9)了解出版详情和人物授权信息。
