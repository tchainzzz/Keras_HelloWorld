from TFInterface import AbstractClassifier
import pandas as pd
import json
import os
import subprocess
import time

default_path = "data/yelp_reviews.csv"
class Yelper(AbstractClassifier):
    def __init__(self, data_path="review", schema="json", report_loaded=10000):

        if os.path.exists(default_path) and os.path.isfile(default_path):
            self.data = pd.read_csv(default_path)
        else:
            print("Counting lines...")
            data_path = "yelp_dataset/{}.{}".format(data_path, schema)            
            num_lines = subprocess.check_output(['wc', '-l', data_path], stderr=subprocess.STDOUT).decode('utf-8').split()[0]
            start_time = time.time()
            with open(data_path, 'r') as f:
                index = 0
                for line in f:
                    json_obj = json.loads(line)
                    keep_cols = ['stars', 'text']
                    if index == 0:
                        df = pd.DataFrame(columns=[k for k in json_obj.keys() if k in keep_cols])
                    df.loc[index] = [v for k,v in json_obj.items() if k in keep_cols]
                    if (index % report_loaded) == (report_loaded - 1):
                        end_time = time.time()
                        print("Loaded {}/{} lines - took {:.4f}s".format(index+1, num_lines, end_time - start_time))
                        start_time = time.time()
                    index += 1
            df.to_csv(default_path)
            self.data = df
        self.data.head(n=20)

    def train(self):
        super().raise_override_error()

    def plot_predictions(self):
        super().raise_override_error()

    def eval(self):
        super().raise_override_error()

    def preprocess(self):
        super().raise_override_error()

    def plot_history(self):
        super().raise_override_error()

    def run_model(self):
        super().raise_override_error()

    def build_model(self):
        super().raise_override_error()

    def preview(self):
        super().raise_override_error()

    def summarize(self):
        super().raise_override_error()


if __name__ == '__main__':
    model = Yelper()
