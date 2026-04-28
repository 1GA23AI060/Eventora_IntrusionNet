# Model Artifacts

Training writes generated artifacts here:

```text
model.h5
ann_baseline.h5
cnn_lstm_model.h5
preprocessor.joblib
metadata.json
metrics.json
```

Run `python model/train.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv` after placing the UNSW-NB15 CSV files in `data/`.
