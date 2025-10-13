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


def predict_and_evaluate(classifier, X, th):
    predicted = classifier.predict_proba(X)[:, 1]
    predicted = (predicted > th).astype(int)
    return predicted


# Make prediction
def make_pred(dt, path_to_file):

    predicted = predict_and_evaluate(model, dt, 0.5)
    
    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': predicted
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission
