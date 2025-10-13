import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, get_best_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)

            logger.info('Making prediction')
            submission, raw_predicted = make_pred(processed_df, file_path)

            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)
            best_features = get_best_features()
            best_features_filename = f"best_features_{timestamp}_{os.path.basename(file_path)}.json"
            with open(os.path.join(self.output_dir, best_features_filename), 'w') as f:
                json.dump(best_features, f)
            logger.info('Best features: %s', best_features_filename)
            
            plt.figure(figsize=(10, 6))
            sns.kdeplot(raw_predicted, fill=True, alpha=0.6)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Distribution of Fraud Prediction Probabilities')
            plt.grid(True, alpha=0.3)
            
            density_plot_filename = f"density_distribution_{timestamp}_{os.path.basename(file_path)}.png"
            plt.savefig(os.path.join(self.output_dir, density_plot_filename), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info('Density distribution plot saved to: %s', density_plot_filename)


        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
