# Dataset Folder

Place UNSW-NB15 CSV files here.

Recommended filenames:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

The training pipeline expects a binary target column named `label`. If extra target/detail columns such as `attack_cat` or `id` exist, they are ignored during feature preparation.

