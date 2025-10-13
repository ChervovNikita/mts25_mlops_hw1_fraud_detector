# Import standard libraries
import pandas as pd
import numpy as np
import logging
import joblib


logger = logging.getLogger(__name__)
RANDOM_STATE = 42

# cat_features = [
#     'merch', 'cat_id', 'gender', 'street', 'one_city', 'us_state', 'post_code', 'jobs'
# ]
num_features = [
    'amount', 'lat', 'lon', 'merchant_lat', 'merchant_lon', 'distance', 'amount_log', 'population_city_log'
]


def process_time(df):
    logger.debug('Adding time features...')
    df = df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['minute'] = df['transaction_time'].dt.minute
    df['time_of_day'] = (df['hour'] + df['minute'] / 60) / 24
    df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'])
    df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'])
    df['is_weekend'] = (df['transaction_time'].dt.dayofweek >= 5).astype(int)
    df = df.drop(columns=['hour', 'minute', 'time_of_day'])
    return df


def process_features(df):
    logger.debug('Processing features...')
    df = df.copy()
    df = process_time(df)
    df['amount_log'] = np.log(df['amount'] + 1)
    df['population_city_log'] = (df['population_city'] - df['population_city'].mean()) / df['population_city'].std()
    df['distance'] = np.sqrt((df['lat'] - df['merchant_lat']) ** 2 + (df['lon'] - df['merchant_lon']) ** 2)
    df.drop(columns=['name_1', 'name_2', 'population_city'], inplace=True)
    return df


# Calculate means for encoding at docker container start
def load_train_data():
    logger.info('Loading training data...')

    train = pd.read_csv('./train_data/train.csv')
    train_x = train.drop(columns=['target'])
    logger.info('Raw train data imported. Shape: %s', train.shape)

    train_x = process_features(train_x)

    scaler = joblib.load('./models/scaler.pkl')
    train_x[num_features] = scaler.transform(train_x[num_features])

    logger.info('Train data processed. Shape: %s', train.shape)

    return train_x


# Main preprocessing function
def run_preproc(train, input_df):
    input_df = process_features(input_df)
    logger.info('Added time features. Output shape: %s', input_df.shape)

    scaler = joblib.load('./models/scaler.pkl')
    input_df[num_features] = scaler.transform(input_df[num_features])
    logger.info('Scaled numerical features. Output shape: %s', input_df.shape)

    return input_df
