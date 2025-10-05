# Objective: Maximize Macro F1-Score
!pip install optuna #for google colab
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)
METRIC = 'f1_macro' 
def load_data():
    data = {
        'age': np.random.randint(29, 77, 303),
        'sex': np.random.randint(0, 2, 303),
        'cp': np.random.randint(0, 4, 303), 
        'trestbps': np.random.randint(90, 200, 303), 
        'chol': np.random.randint(120, 564, 303), 
        'fbs': np.random.randint(0, 2, 303), 
        'restecg': np.random.randint(0, 3, 303),
        'thalach': np.random.randint(71, 202, 303), 
        'exang': np.random.randint(0, 2, 303), 
        'oldpeak': np.random.rand(303) * 6.2,
        'slope': np.random.randint(0, 3, 303),
        'ca': np.random.randint(0, 4, 303),
        'thal': np.random.randint(1, 4, 303),
        'target': np.random.randint(0, 2, 303)
    }
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    preproc = ColumnTransformer(
        transformers= [
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
        remainder='passthrough'#using remainder = 'drop' would be better memory wise
    )    
    return X_train, X_test, y_train, y_test, preproc
def objective(trial: optuna.Trial, X_train, y_train, preproc):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.6, 1.0),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        **params
    )
    pipe = Pipeline(steps=[('preproc', preproc), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    score = cross_val_score(
        pipe, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring=METRIC, 
        n_jobs=-1
    ).mean()
    return score
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preproc = load_data()
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler
    )
    print("Starting BO (TPE) optimization...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preproc), 
        n_trials=100 #i worked on n_trails = 10 as 100 was taking a long time and each trial does 5 cross validations on a random forest which is computationally heavy
    )
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Best CV F1-Score: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    best_params = study.best_params
    final_model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_params)
    final_pipe = Pipeline(steps=[('preproc', preproc), ('model', final_model)])
    final_pipe.fit(X_train, y_train)
    test_f1 = f1_score(y_test, final_pipe.predict(X_test), average='macro')
    test_acc = final_pipe.score(X_test, y_test)
    print("\n--- Final Test Score ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    #if necessary, can print runtime per trial for progress feedback, by importing time 
