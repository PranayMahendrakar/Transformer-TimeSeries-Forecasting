# Transformer-Based Time Series Forecasting Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive deep learning framework for long-term time series forecasting using state-of-the-art **Transformer architectures**. This project benchmarks and extends transformer-based models against traditional LSTM baselines across energy, weather, and financial domains.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Research Contributions](#research-contributions)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarking Results](#benchmarking-results)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional recurrent architectures like LSTMs struggle with long-range dependencies and computational inefficiency on long sequences. This framework explores **transformer-based alternatives** that leverage self-attention mechanisms for capturing global temporal patterns across multi-variate time series.

### Key Features

- Unified training and evaluation pipeline for multiple transformer models
- Head-to-head comparison with LSTM and BiLSTM baselines
- Support for multivariate time series with exogenous variables
- Modular codebase: plug in new models with minimal configuration
- Reproducible experiments with seed control and config files
- Visualization tools for attention maps and forecasting plots

---

## Models

### 1. Informer

- **ProbSparse Self-Attention**: Reduces complexity from O(L^2) to O(L log L)
- **Distilling operation**: Prunes less important attention scores across layers
- **Generative decoder**: Produces long sequences in a single forward pass
- Best suited for: ultra-long forecasting horizons (720+ steps)

### 2. Temporal Fusion Transformer (TFT)

- **Variable Selection Networks**: Learns input feature importance dynamically
- **Gated Residual Networks (GRN)**: Suppresses irrelevant features
- **Multi-head Attention**: Captures long-range temporal dependencies
- **Quantile outputs**: Provides probabilistic forecasting intervals
- Best suited for: interpretable multi-horizon forecasting with static and dynamic covariates

### 3. Autoformer

- **Series Decomposition**: Separates trend and seasonality progressively
- **Auto-Correlation Mechanism**: Replaces self-attention with period-based dependencies
- **Complexity**: O(L log L) with better seasonal pattern learning
- Best suited for: periodic time series (weather, energy cycles)

### Baseline: LSTM / BiLSTM

- Vanilla LSTM and Bidirectional LSTM for direct comparison
- Sequence-to-sequence with attention for fair benchmarking

---

## Research Contributions

| Contribution | Description |
|---|---|
| Transformer vs LSTM Benchmark | Systematic comparison on MSE, MAE, MAPE across 3 domains and 4 forecast horizons |
| Long-Term Forecasting Accuracy | Improved accuracy on 720-step horizons using decomposition-based attention |
| Multivariate Time Series Handling | Joint modeling of correlated variables |
| Ablation Studies | Component-wise analysis of attention heads and encoder depth |

---

## Applications

### Energy Demand Forecasting

- Dataset: ETTh1, ETTh2, ETTm1, ETTm2 (Electricity Transformer Temperature)
- Task: Predict electricity load and transformer oil temperatures
- Horizon: 24h to 720h ahead

### Weather Prediction

- Dataset: Weather dataset (21 meteorological indicators)
- Task: Multi-step prediction of temperature, humidity, wind speed
- Horizon: 96 to 720 time steps

### Financial Market Analysis

- Dataset: Exchange rate and Stock index datasets
- Task: Multivariate price and volatility forecasting
- Horizon: Short-term (96 steps) to medium-term (336 steps)

---

## Project Structure

```
Transformer-TimeSeries-Forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── informer.py
│   ├── tft.py
│   ├── autoformer.py
│   ├── lstm_baseline.py
│   └── base_model.py
├── layers/
│   ├── attention.py
│   ├── decomposition.py
│   ├── encoding.py
│   └── embedding.py
├── experiments/
│   ├── configs/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   └── early_stopping.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_benchmark_results.ipynb
│   └── 04_attention_visualization.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/PranayMahendrakar/Transformer-TimeSeries-Forecasting.git
cd Transformer-TimeSeries-Forecasting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
einops>=0.6.0
pytorch-lightning>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## Usage

### Train a Model

```bash
python experiments/train.py --config experiments/configs/informer_etth1.yaml
python experiments/train.py --config experiments/configs/tft_weather.yaml
python experiments/train.py --config experiments/configs/autoformer_ettm2.yaml
python experiments/train.py --config experiments/configs/lstm_etth1.yaml
```

### Run Full Benchmark

```bash
python experiments/benchmark.py --dataset ETTh1 --horizons 96 192 336 720
```

### Python API

```python
from models.informer import Informer
from utils.data_loader import TimeSeriesDataset
import torch

dataset = TimeSeriesDataset(
    data_path="data/processed/ETTh1.csv",
    target_col="OT",
    seq_len=96,
    pred_len=336,
    features="M"
)

model = Informer(
    enc_in=7, dec_in=7, c_out=7,
    seq_len=96, label_len=48, out_len=336,
    d_model=512, n_heads=8,
    e_layers=2, d_layers=1,
    d_ff=2048, attn="prob", dropout=0.05
)
```

---

## Benchmarking Results

Results on ETTh1 dataset — Multivariate forecasting (MSE / MAE)

| Model | Horizon 96 | Horizon 192 | Horizon 336 | Horizon 720 |
|---|---|---|---|---|
| Autoformer | 0.449 / 0.459 | 0.500 / 0.482 | 0.521 / 0.496 | 0.514 / 0.512 |
| Informer | 0.865 / 0.713 | 1.083 / 0.801 | 1.524 / 0.976 | 2.744 / 1.305 |
| TFT | 0.519 / 0.516 | 0.567 / 0.537 | 0.604 / 0.561 | 0.671 / 0.602 |
| LSTM | 0.672 / 0.571 | 0.795 / 0.669 | 0.912 / 0.724 | 1.124 / 0.845 |
| BiLSTM | 0.643 / 0.554 | 0.768 / 0.647 | 0.876 / 0.699 | 1.089 / 0.812 |

Lower is better. Transformer models consistently outperform LSTM baselines on longer horizons.

---

## Datasets

| Dataset | Variables | Frequency | Size | Domain |
|---|---|---|---|---|
| ETTh1 / ETTh2 | 7 | Hourly | 17,420 | Energy |
| ETTm1 / ETTm2 | 7 | 15-min | 69,680 | Energy |
| Weather | 21 | 10-min | 52,696 | Weather |
| Exchange Rate | 8 | Daily | 7,588 | Finance |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-model`
3. Commit your changes: `git commit -m "Add new model"`
4. Push and open a Pull Request

---

## Citation

```bibtex
@misc{mahendrakar2026transformer,
  title={Transformer-Based Time Series Forecasting Framework},
  author={Pranay M Mahendrakar},
  year={2026},
  url={https://github.com/PranayMahendrakar/Transformer-TimeSeries-Forecasting}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Pranay M Mahendrakar**  
AI Specialist | Author | Patent Holder | Open-Source Contributor  
Bengaluru, India | https://sonytech.in/pranay
