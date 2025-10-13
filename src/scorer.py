import pandas as pd
import logging
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/catboost_model.cbm')
logger.info('Pretrained model imported successfully...')


def get_best_features():
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(5)
    return {k: v for k, v in zip(importance_df.feature, importance_df.importance)}


def predict_and_evaluate(classifier, X, th):
    raw_predicted = classifier.predict_proba(X)[:, 1]
    predicted = (raw_predicted > th).astype(int)
    return predicted, raw_predicted


# Make prediction
def make_pred(dt, path_to_file):
    predicted, raw_predicted = predict_and_evaluate(model, dt, 0.5)
    
    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': predicted
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission, raw_predicted
