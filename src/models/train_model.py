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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


###################################################
# define training
def training(FILE, callback=None):
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
    y_train_np = y_train.values # need to convert for autologging
    y_test_np = y_test.values
    logger.info("train-test split is done.")

    # Define baseline models
    params = {
        "KNeighbors": {
            "n_neighbors": 25,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
        },
        "DecisionTree": {
            "max_depth": 6,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        "RandomForest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        "GradientBoosting": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
        },
    }

    models = {
        "KNeighbors": KNeighborsClassifier(**params["KNeighbors"]),
        "DecisionTree": DecisionTreeClassifier(**params["DecisionTree"]),
        "RandomForest": RandomForestClassifier(**params["RandomForest"]),
        "GradientBoosting": GradientBoostingClassifier(**params["GradientBoosting"]),
        }

    first_model = True
    total_model = len(models)
    i = 0
    for name, model in models.items():
        if callback:
            callback(20 + int((90-20)/total_model * i),
                     f"Training {name}...")
            i += 1

        model.fit(X_train_np, y_train_np)
        y_pred = model.predict(X_test_np)

        acc = accuracy_score(y_test_np, y_pred)
        prec = precision_score(y_test_np, y_pred)
        rec = recall_score(y_test_np, y_pred)
        f1 = f1_score(y_test_np, y_pred)

        logger.info(f"\nModel: {name}")
        logger.info(f"  Accuracy : {acc}")
        logger.info(f"  Precision: {prec}")
        logger.info(f"  Recall   : {rec}")
        logger.info(f"  F1-score : {f1}")

        if first_model:
            best_name, best_acc, best_prec, best_rec, best_f1, best_model = name, acc, prec, rec, f1, model
            first_model = False
        else:
            # Choose best model by F1-score
            if f1 > best_f1:
                best_name, best_acc, best_prec, best_rec, best_f1, best_model = name, acc, prec, rec, f1, model

    if callback:
        callback(90, "Logging best model...")

    with mlflow.start_run(run_name="weather_20percent_best_model") as run:
        mlflow.log_params(params[best_name])
        # log best model metrics
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
        mlflow.set_tag("Training Info", f"saved {best_name} as the best model.")
        client = MlflowClient()
        try:
            # fetch the champion model
            version = client.get_model_version_by_alias(name=model_name, alias="champion")
            production_f1 = client.get_run(version.run_id).data.metrics.get("f1_score", 0)
            # check if the trained model outperforms the champion model
            if best_f1 > production_f1:
                logger.info(f"promoting new model to champion, new best f1: {best_f1}, old best f1: {production_f1}")
                # promote the new current best model to champion
                client.set_registered_model_alias(name=model_name, alias="champion", version=model_info.registered_model_version)
        except:
            # if no champion model exists, create one
            client.set_registered_model_alias(name=model_name, alias="champion", version=model_info.registered_model_version)

    logger.info("best model is saved.")

    logger.info("\nBest model (by F1-score):")
    logger.info(f"  Name     : {best_name}")

    joblib.dump(list(X.columns), FEATURES_PATH)
    logger.info("training is finished.")

    if callback:
        callback(100, log_stream.getvalue())

#####################################################


