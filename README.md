# 🚬 Smoking Status Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](#)

A machine learning project to predict whether an individual is a smoker based on medical and lifestyle data using **XGBoost** , feature engineering, hyperparameter tuning, and ensemble methods.

This project was developed as part of the Data Mining course at National Cheng Kung University (NCKU) , and achieved 2nd place in the class competition.

![Leaderboard](/Private_leaderboard.png)

---

## 📁 Dataset Description

The dataset includes health-related features from individuals and the target variable indicating **smoking status** (1 = smoker, 0 = non-smoker). The following files are used:

|File|	Description|	Source|
|--|--|--|
|train.csv|	Base training data provided by the course|	[NCKU CS Data Mining Homework 2](https://www.kaggle.com/competitions/ncku-cs-data-mining-homework-2/data)|
|train_Medium.csv|	Additional synthetic training data for model improvement|	[Smoker Status Prediction Using Biosignals Dataset](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)|
|train_Large.csv|	Extra large-scale training data from a related competition|	[Playground Series S3E24](https://www.kaggle.com/c/playground-series-s3e24/data)|
|test.csv|	Test set for final submission|	[NCKU CS Data Mining Homework 2](https://www.kaggle.com/competitions/ncku-cs-data-mining-homework-2/data)|

### Sample Features:
| Feature | Description |
|--------|-------------|
| `age` | Age in years |
| `gender` | Gender (Male/Female) |
| `height(cm)` | Height in centimeters |
| `weight(kg)` | Weight in kilograms |
| `systolic`, `relaxation` | Blood pressure readings |
| `HDL`, `LDL`, `triglyceride` | Lipid profile |
| `AST`, `ALT`, `Gtp` | Liver enzyme levels |
| `Urine protein` | Protein levels in urine |
| `hearing(left/right)` | Hearing test result |
| `smoking` | Target label |

---

## Project Highlights

- **Feature Engineering**: Created new meaningful features like BMI, LDL/HDL ratio, AST/ALT ratio, etc.
- **Outlier Handling**: Cap outliers at ±3 standard deviations
- **Hyperparameter Tuning**: Used `RandomizedSearchCV` to find optimal XGBoost parameters
- **Cross-Validation**: Trained models using 10-fold Stratified K-Fold CV
- **Ensemble Learning**: Combined predictions from multiple models using Optuna-optimized weights

---

## Requirements

Make sure you have installed the following libraries:

```bash
pip install pandas numpy xgboost scikit-learn optuna seaborn matplotlib
```

---

## File Structure

```
.
├── README.md
├── main.py
├── train.csv
├── train_Medium.csv
├── train_Large.csv
└── test.csv
```

Place all CSV files in the same directory as the script.

---

## How to Run

1. Clone the repo:

```bash
git clone https://github.com/yourusername/smoking-prediction.git
cd smoking-prediction
```

2. Place the dataset files (`train.csv`, `train_Medium.csv`, `train_Large.csv`, `test.csv`) in the root folder.

3. Run the script:

```bash
python main.py
```

4. A file named `submission.csv` will be generated containing predicted probabilities for the test set.

---

## Results

During execution, the script outputs:

- Best hyperparameters found during tuning
- Validation AUC per fold
- Average and best AUC across folds
- Final submission file saved as `submission.csv`

---

## Example Output

```
Fold 1 | AUC: 0.876543
Fold 2 | AUC: 0.878901
...
ALL fold average AUC: 0.8774  Best AUC: 0.8812
Best weights: {'w1': 0.42, 'w2': 0.35, 'w3': 0.23}
Submission file saved as submission.csv
```

---

## Acknowledgments

This project was inspired by real-world applications of predictive modeling in healthcare and preventive medicine.

---

## License

This project is licensed under the [MIT License](LICENSE).
