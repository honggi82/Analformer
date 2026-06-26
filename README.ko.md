# Analformer: EEG 디코딩 및 신경과학 분석을 위한 새로운 변환기 아키텍처

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md"><strong>한국어</strong></a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md">中文</a>
</div>

이 저장소는 Scientific Reports 기사의 공식 구현을 제공합니다.

**EEG 디코딩 및 신경과학 분석을 위한 새로운 변환기 아키텍처**

염홍기, 최우성, 안경민, *과학 보고서*(2026).

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer는 고성능 BCI(뇌-컴퓨터 인터페이스) 디코딩과 해석 가능한 신경과학 분석을 모두 지원하도록 설계된 Transformer 기반 아키텍처입니다. 이 모델은 Analytical Patch Embedding 모듈에서 훈련이 불가능한 고정된 Morlet 웨이블릿 커널을 사용하므로 시간-주파수 맵, 지형, F-값 시간-주파수(FTF) 분석 및 주의 기반 기능 연결과 같은 친숙한 EEG 분석 도구를 사용하여 중간 표현을 분석할 수 있습니다.

## 개요

Analformer는 네 가지 디코딩 설정을 포함하는 공개 EEG 데이터 세트에서 평가되었습니다.

- **BCI Competition IV 2a**: 4가지 클래스 MI(운동 이미지)
- **OpenBMI MI**: 2클래스 모터 이미지
- **OpenBMI ERP**: 2클래스 이벤트 관련 잠재적 디코딩
- **OpenBMI SSVEP**: 4등급 정상 상태 시각적 유발 전위 디코딩

핵심 아이디어는 임베딩 중에 해석 가능한 EEG 구조를 보존하는 것입니다. 고정 Morlet 웨이블릿 필터는 시공간 주파수 특징을 추출하고, Transformer 인코더는 이러한 특징 간의 관계를 학습하며, 분류 헤드는 목표 BCI 클래스를 예측합니다. 분석 출력은 모델의 내부 표현과 주의 가중치를 통해 생성됩니다.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## 저장소 구조

- `Analformer_BCI_Comp_4_2a.ipynb`: BCI 경쟁 IV 2a 모터 이미지 구현.
- `Analformer_OpenBMI_MI.ipynb`: OpenBMI 모터 이미지 구현.
- `Analformer_OpenBMI_ERP.ipynb`: OpenBMI ERP 구현.
- `Analformer_OpenBMI_SSVEP.ipynb`: OpenBMI SSVEP 구현.
- `figures/Graphical_abstract.png`: Analformer의 그래픽 요약.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png`: 논문의 분석 결과 예입니다.
- `requirements.txt`: 노트북에서 사용되는 Python 패키지 종속성입니다.

## 기술 요구 사항

Python 3.9+를 권장합니다. 교육에는 CUDA 지원 PyTorch 설치가 권장됩니다.

CUDA 환경에 따라 먼저 PyTorch를 설치하세요. 예를 들면:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

그런 다음 나머지 종속성을 설치합니다.

```bash
pip install -r requirements.txt
```

## 데이터 준비

데이터 세트는 이 저장소에 포함되지 않습니다. 원본 소스에서 공개 데이터세트를 다운로드하세요.

- BCI 대회 IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- 오픈BMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

각 노트북에는 `data` 및 `label` 어레이가 포함된 HDF5 호환 `.mat` 파일이 필요합니다. 노트북은 다음 이름 지정 패턴을 사용하여 주제 파일을 로드합니다.

- 훈련 파일: `A1T.mat`, `A2T.mat`, ...
- 평가 파일: `A1E.mat`, `A2E.mat`, ...

`main()` 내부의 `params` 사전에 데이터세트 폴더를 설정합니다.

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

코드는 250Hz EEG 데이터를 가정합니다. OpenBMI 노트북은 62개 채널을 사용하는 반면, BCI Competition IV 2a 노트북은 22개 채널을 사용합니다.

## 사용방법

1. 실행하려는 데이터 세트/패러다임에 대한 노트북을 엽니다.
2. `params` 사전의 `"root"`를 로컬 데이터세트 폴더로 업데이트합니다.
3. `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads`, `att_ch` 등 주요 실험 매개변수를 확인하세요.
4. 노트북 셀을 순서대로 실행합니다.
5. `"anal": 1`를 사용하여 시간-주파수 맵, 지형, F 값 맵 및 주의 기반 연결 시각화를 포함한 분석 출력을 생성합니다.

OpenBMI 노트북의 경우 주제별 미세 조정 전에 `Pretraining(params)`가 실행됩니다. BCI Competition IV 2a 노트북에서 현재 워크플로는 사전 훈련을 건너뛸 때 사전 훈련된 체크포인트 경로를 사용합니다. 해당 노트북을 실행하기 전에 `pretrained_path`를 업데이트하세요.

## 분석 결과 예시

다음 그림은 논문의 대표적인 분석 결과 예를 보여줍니다.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## 인용

이 코드를 사용하는 경우 다음을 인용해 주세요.

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## 감사의 말

본 연구는 조선대학교 연구비의 지원을 받았습니다.

이 구현의 일부는 GPL-3.0 라이센스에 따라 배포되는 [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) 저장소를 참조하여 개발되었습니다. 구현을 공개적으로 제공한 작성자에게 감사드립니다.

## 라이센스

소스 코드는 `LICENSE` 라이선스에 따라 배포됩니다. 종이 그림은 공식적인 구현을 설명하기 위해 여기에 재현되었습니다. 출판 세부 사항 및 피규어 라이센스 정보는 [기사 페이지](https://www.nature.com/articles/s41598-026-56405-9)를 참조하세요.
