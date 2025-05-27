import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler



def feature_engineering(df):
    # BMI
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    # Blood_Pressure
    df['Blood_Pressure'] = df['systolic'] / df['relaxation']
    # LDL / HDL
    df['LDL_to_HDL'] = df['LDL'] / df['HDL']
    # AST / ALT
    df['AST_ALT_ratio'] = df['AST'] / df['ALT']
    # Features interaction
    df['height_weight'] = df['height(cm)'] * df['weight(kg)']
    df['waist_triglyceride'] = df['waist(cm)'] * df['triglyceride']
    df['BMI_ALT'] = df['BMI'] * df['ALT']
    df['creatinine_hemoglobin'] = df['serum creatinine'] * df['hemoglobin']
    df['systolic_diastolic'] = df['systolic'] * df['relaxation']
    # One Hot encoding
    df[['hearing(left)', 'hearing(right)', 'Urine protein']] = df[['hearing(left)', 'hearing(right)', 'Urine protein']].replace({2: 0})

    return df

def outliers_processing(data, category_cols):
    data_cleaned = data.copy()
    for col in category_cols:
        mean = data[col].mean()
        std = data[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        outlier_upper_bound_mask = (data[col] > upper_bound)
        outlier_lower_bound_mask = (data[col] < lower_bound)
        data_cleaned.loc[outlier_upper_bound_mask, col] = upper_bound
        data_cleaned.loc[outlier_lower_bound_mask, col] = lower_bound
    return data_cleaned


def xgb_param_RandomSearch(X, y):
    # Define the hyperparameter range
    param_dist = {
        'n_estimators': randint(1000, 1500),
        'max_depth': randint(2, 8),
        'learning_rate': uniform(0.01, 0.2),
        'reg_lambda': uniform(0, 1),
        'reg_alpha': uniform(0, 1),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': uniform(1, 100),
    }

    # Base XGBoost model
    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist'
    )

    print("Starting hyperparameter tuning using RandomizedSearchCV...\n")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30, # setting iterations
        scoring='roc_auc',
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X, y)
    best_params = random_search.best_params_

    # Update base model param
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist'
    })

    print("\nBest parameters found:")
    print(best_params)
    print(f"Best AUC from tuning: {random_search.best_score_:.6f}\n")

    return best_params, random_search


# Cross-validation training using the best parameters
def xgboost_kfold_fit(model_params, kf, X, y, final_test):
    
    session = {
        'scores': [],
        'predictions': [],
        'models': [],
        'oof': []
    }
    print(f"Training XGBoost Models with Cross Validation")
    print('----------------------------------')
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Split the data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create and fit model
        model = XGBClassifier(**model_params)
        model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_val, y_val)],
            verbose=0
        )

        # Predict on validation and test sets
        val_predictions = model.predict_proba(X_val)[:, 1]
        # performe one hot encoding on predict porba by threshold=0.5
        val_predictions[val_predictions <= 0.5] = 0
        val_predictions[val_predictions > 0.5] = 1
        val_score = roc_auc_score(y_val, val_predictions)
        test_predictions = model.predict_proba(final_test)[:, 1]

        # Record results
        session['scores'].append(val_score)
        session['predictions'].append(test_predictions)
        session['models'].append(model)
        session['oof'].append(val_predictions)

        print(f"Fold {fold + 1} | AUC: {val_score:.6f}")

    print('==================================')
    print(f"ALL fold average AUC: {np.mean(session['scores']):.6f}  Best AUC: {np.max(session['scores']):.6f}")
    print('==================================')
    
    return session

def Search_Bestweight(model1, model2, model3, X, y, kf):
    # Get out-of-fold(oof)
    model1_oof = np.zeros(len(y))
    model2_oof = np.zeros(len(y))
    model3_oof = np.zeros(len(y))
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        model1_oof[val_idx] = model1['oof'][i]
        model2_oof[val_idx] = model2['oof'][i]
        model3_oof[val_idx] = model3['oof'][i]
    
    #object function for optuna
    def objective(trial):
        w1 = trial.suggest_float('w1', 0, 1)
        w2 = trial.suggest_float('w2', 0, 1)
        w3 = trial.suggest_float('w3', 0, 1)
        
        total = w1 + w2 + w3
        w1 /= total
        w2 /= total
        w3 /= total
        
        y_pred = w1 * model1_oof + w2 * model2_oof + w3 * model3_oof
        return roc_auc_score(y_all, y_pred)
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100)

    best_weights = study.best_params
    total_weight = best_weights['w1'] + best_weights['w2'] + best_weights['w3']
    best_weights['w1'] /= total_weight
    best_weights['w2'] /= total_weight
    best_weights['w3'] /= total_weight 
    print("Best weights:", best_weights)
    
    return best_weights


if __name__ == "__main__":
    # Load data
    path = "./"
    train_path = path + "train.csv"
    train_l_path = path + "train_Large.csv"
    train_M_path = path + "train_Medium.csv"
    test_path = path + "test.csv"
    train_orig_data = pd.read_csv(train_path)
    train_l_data = pd.read_csv(train_l_path)
    train_m_data = pd.read_csv(train_M_path)
    test_data = pd.read_csv(test_path)

    # Drop ID columns
    feature_m_data = train_m_data.drop(columns=["id"])
    feature_l_data = train_l_data.drop(columns=["id"])
    feature_orig_data = train_orig_data.drop(columns=["id"])
    final_test_data = test_data.drop(columns=["id"])

    # Feature engineering
    feature_l_data = feature_engineering(feature_l_data)
    feature_m_data = feature_engineering(feature_m_data)
    feature_orig_data = feature_engineering(feature_orig_data)
    final_test_data = feature_engineering(final_test_data)

    # Split features and target
    X_l = feature_l_data.drop(columns=["smoking"])
    y_l = feature_l_data["smoking"]
    X_orig = feature_orig_data.drop(columns=["smoking"])
    y_orig = feature_orig_data["smoking"]
    X_m = feature_m_data.drop(columns=["smoking"])
    y_m = feature_m_data["smoking"]
    
    number_cols = ['systolic','relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'AST','ALT', 'Gtp', 'dental caries', 'BMI', 'Blood_Pressure',
       'LDL_to_HDL', 'AST_ALT_ratio', 'height_weight', 'waist_triglyceride',
       'BMI_ALT', 'creatinine_hemoglobin', 'systolic_diastolic' ]
    # Replace outlier by upper bound or lower bound
    X_l_clean = outliers_processing(X_l, number_cols)
    X_m_clean = outliers_processing(X_m, number_cols)
    X_orig_clean = outliers_processing(X_orig, number_cols)
    
    # Search the best parameter for 3 different size data
    param_from_X_orig, X_orig_random_search_res = xgb_param_RandomSearch(X_orig_clean, y_orig)
    param_from_X_m, X_m_random_search_res = xgb_param_RandomSearch(X_m_clean, y_m)
    param_from_X_l, X_l_random_search_res = xgb_param_RandomSearch(X_l_clean, y_l)
    
    kf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Concat all dataset
    X_all = pd.concat([X_orig_clean, X_m_clean, X_l_clean])
    y_all = pd.concat([y_orig, y_m, y_l])
    
    # Training 3 models by 3 different best parameters
    xgboost_m1 = xgboost_kfold_fit(param_from_X_orig, kf_10, X_all, y_all, final_test_data)
    xgb_m1_test_avg = np.mean(np.array(xgboost_m1['predictions']), axis=0).ravel()

    xgboost_m2 = xgboost_kfold_fit(param_from_X_m, kf_10, X_all, y_all, final_test_data)
    xgb_m2_test_avg = np.mean(np.array(xgboost_m2['predictions']), axis=0).ravel()

    xgboost_m3 = xgboost_kfold_fit(param_from_X_l, kf_10, X_all, y_all, final_test_data)
    xgb_m3_test_avg = np.mean(np.array(xgboost_m3['predictions']), axis=0).ravel()
    
    best_weights = Search_Bestweight(xgboost_m1, xgboost_m2, xgboost_m3, X_all, y_all, kf_10)
    test_preds = xgb_m1_test_avg*best_weights['w1'] + xgb_m2_test_avg*best_weights['w2'] + xgb_m3_test_avg*best_weights['w3']

    # Create submission data
    submission = pd.DataFrame({
        'id': test_data['id'],
        'smoking': test_preds
    })

    # Export predictions to CSV
    save_name = 'submission.csv'
    submission.to_csv(save_name, index=False)
    print(f"Submission file saved as {save_name}")
