# AI-Based Network Intrusion Detection System using CNN-LSTM

Full-stack intrusion detection project using the UNSW-NB15 dataset, TensorFlow/Keras models, a Flask API, and an analyst-style HTML/CSS/JavaScript dashboard.

## Project Structure

```text
backend/
  app.py
  config.py
  inference.py
frontend/
  index.html
  styles.css
  app.js
model/
  train.py
  preprocess.py
  artifacts/
data/
  README.md
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add Dataset

Download the UNSW-NB15 CSV files and place them in `data/`.

Supported names include:

```text
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

You can also pass custom paths to the training command.

## Train Models

```bash
python model/train.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv --epochs 25 --min-precision 0.90
```

This trains:

- ANN baseline model
- Advanced CNN-BiLSTM-attention hybrid model

The training pipeline applies robust scaling, rare-category handling, variance filtering, class balancing, PR-AUC monitoring, learning-rate reduction, early stopping, and precision-aware threshold tuning.

Outputs are saved to `model/artifacts/`:

- `model.h5`
- `preprocessor.joblib`
- `metadata.json`
- `metrics.json`
- `ann_baseline.h5`
- `cnn_lstm_model.h5`

`metrics.json` includes accuracy, precision, recall, F1 score, ROC-AUC, average precision, confusion matrix, classification report, and the selected decision threshold.

## Run API and Frontend

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API

Health check:

```bash
curl http://127.0.0.1:5000/
```

Single prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"dur\":0.1,\"proto\":\"tcp\",\"service\":\"http\",\"state\":\"FIN\"}}"
```

CSV upload prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict -F "file=@data/sample.csv"
```

The response includes `Normal` or `Attack` predictions, severity, attack probability, attack rate, average risk, high-confidence alert count, and dashboard counts.
