import os
import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import logging
from io import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


###################################################
# define training
def training(traning_args, callback=None):
    FILE = traning_args['processed_data_file']
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(THIS_DIR, "../../models")
    FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

    log_stream = StringIO()
    logger = logging.getLogger("train_model")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    # initialize mlflow experiment; allow override via env for Docker
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"))
    experiment = mlflow.get_experiment_by_name("MLflowTrackingWeatherAustralia")
    if experiment is None:
        experiment_id = mlflow.create_experiment("MLflowTrackingWeatherAustralia")
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    # mlflow.sklearn.autolog()

    if callback:
        callback(10, "Loading data...")

    df = pd.read_csv(FILE)

    logger.info("data is loaded.")

    # drop index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    y = df["RainTomorrow"].astype(int)   
    X = df.drop(columns=["RainTomorrow"])

    if callback:
        callback(20, "Splitting data...")

    # Splot dataset into test aund train set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    logger.info("train-test split is done.")

    # define cross-validation parameters
    cv_split = StratifiedKFold(n_splits=3, shuffle=True)
    n_iter_search = 6

    # define parameter grids for each model
    param_grids = {
        "KNeighbors": {
            "n_neighbors": [5, 15, 25, 35],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "DecisionTree": {
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 10, 15, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "subsample": [0.7, 0.8, 1.0],
        },
    }

    base_models = {
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    first_model = True
    best_params = {}
    model_list = list(base_models.keys())
    total_models = len(model_list)

    for i, name in enumerate(model_list):
        if callback:
            callback(
                20 + int((90 - 20) / total_models * i),
                f"Training {name}...",
            )

        # use random search for each model to find the best parameters according to the f1 score
        search = RandomizedSearchCV(
            base_models[name],
            param_grids[name],
            n_iter=n_iter_search,
            cv=3,
            scoring="f1",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train_np, y_train_np)
        model = search.best_estimator_
        best_params[name] = search.best_params_

        cv_scores = cross_val_score(
            model, X_train_np, y_train_np, cv=cv_split, scoring="f1"
        )
        mean_cv_f1 = float(cv_scores.mean())
        logger.info(f"\nModel: {name} (Mean CV F1: {mean_cv_f1:.4f})")

        y_pred = model.predict(X_test_np)
        acc = accuracy_score(y_test_np, y_pred)
        prec = precision_score(y_test_np, y_pred, zero_division=0)
        rec = recall_score(y_test_np, y_pred, zero_division=0)
        f1 = f1_score(y_test_np, y_pred, zero_division=0)

        logger.info(f"  Accuracy : {acc}")
        logger.info(f"  Precision: {prec}")
        logger.info(f"  Recall   : {rec}")
        logger.info(f"  F1-score : {f1}")

        if first_model:
            best_name, best_acc, best_prec, best_rec, best_f1, best_mean_cv_f1, best_model = (
                name,
                acc,
                prec,
                rec,
                f1,
                mean_cv_f1,
                model,
            )
            first_model = False
        elif mean_cv_f1 > best_mean_cv_f1:
            best_name, best_acc, best_prec, best_rec, best_f1, best_mean_cv_f1, best_model = (
                name,
                acc,
                prec,
                rec,
                f1,
                mean_cv_f1,
                model,
            )

    if callback:
        callback(90, "Logging best model...")

    with mlflow.start_run(run_name="weather_model") as run:
        # log best model metrics and parameters
        mlflow.log_params(best_params.get(best_name, {}))
        mlflow.log_metric("mean_cv_f1", best_mean_cv_f1)
        mlflow.log_metric("accuracy", best_acc)
        mlflow.log_metric("precision", best_prec)
        mlflow.log_metric("recall", best_rec)
        mlflow.log_metric("f1_score", best_f1)
        # log best model
        model_name = "best_AUS_weather_model"
        model_info = mlflow.sklearn.log_model(sk_model=best_model,
                                              name=f"best_model_{best_name.lower()}",
                                              input_example=X_train_np[:1],
                                              registered_model_name=model_name)
        mlflow.set_tag("best_model_name", f"{best_name}")
        mlflow.set_tag("sample_percent", f"{traning_args['sample_percent']}")
        mlflow.set_tag("duration", f"{traning_args['duration']} years")
        client = MlflowClient()
        try:
            # fetch the champion model
            version = client.get_model_version_by_alias(name=model_name, alias="champion")
            production_mean_cv_f1 = client.get_run(version.run_id).data.metrics.get("mean_cv_f1", 0)
            # check if the trained model outperforms the champion model
            if best_mean_cv_f1 > production_mean_cv_f1:
                logger.info(f"promoting new model to champion, new best f1: {best_mean_cv_f1}, old best f1: {production_mean_cv_f1}")
                # promote the new current best model to champion
                client.set_registered_model_alias(name=model_name, alias="champion", version=model_info.registered_model_version)
        except:
            # if no champion model exists, create one
            client.set_registered_model_alias(name=model_name, alias="champion", version=model_info.registered_model_version)

    logger.info("best model is saved.")

    logger.info("\nBest model (by Mean CV F1):")
    logger.info(f"  Name     : {best_name}")

    joblib.dump(list(X.columns), FEATURES_PATH)
    logger.info("training is finished.")

    if callback:
        callback(100, log_stream.getvalue())

#####################################################


