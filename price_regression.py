from TFInterface import AbstractClassifier
from TFCustomUtils import PeriodicEpochLogger
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PriceRegressionModel(AbstractClassifier):
    def __init__(self):

        def norm(x):
            train_stats = self.data.describe().transpose()
            return (x - train_stats['mean']) / train_stats['std']

        dataset_path = keras.utils.get_file(
            "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                        'Acceleration', 'Model Year', 'Origin']
        self.data = pd.read_csv(dataset_path, names=column_names,
                                na_values="?", comment='\t',
                                sep=" ", skipinitialspace=True).dropna()
        origin = self.data.pop('Origin')
        self.data['USA'] = (origin == 1) * 1.0
        self.data['Europe'] = (origin == 2) * 1.0
        self.data['Japan'] = (origin == 3) * 1.0
        self.train_data = norm(self.data.sample(frac=0.8, random_state=0))
        self.test_data = norm(self.data.drop(self.train_data.index))
        self.train_labels = self.train_data.pop('MPG')
        self.test_labels = self.test_data.pop('MPG')
        self.model = None

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(self.train_data.keys())]),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])


    def run_model(self, **kwargs):
        super().model_check()
        self.train(**kwargs)
        self.eval()

    def preview(self):
        sns.pairplot(
            self.train_data[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")  
        plt.show()

    def eval(self):
        loss, mae, mse = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        print("Test Mean Abs Error: {:8.4f} MPG".format(mae))
        print("Test Mean Sq Error: {:8.4f} MPG^2".format(mse))

    def train(self, n_epochs=1000, val_prop=0.2, verbose=False, logging_frequency=50,
            check_every=10):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=check_every)
        self.history = self.model.fit(self.train_data, self.train_labels,
                            epochs=n_epochs, validation_split = val_prop, 
                            verbose=int(verbose), 
                            callbacks=[PeriodicEpochLogger(
                                        frequency=logging_frequency,
                                        logging_params=['loss', 'mean_absolute_error', 
                                        'mean_squared_error','val_loss']),
                                        early_stop
                                    ])


    def preprocess(self):
        super().raise_override_error()

    def plot_predictions(self):
        super().model_check()
        test_predictions = self.model.predict(self.test_data).flatten()

        plt.scatter(self.test_labels, test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        plt.show()

    def plot_history(self, plt_1_top=5, plt_2_top=20):
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                    label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                    label = 'Val Error')
        plt.ylim([0,plt_1_top])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                    label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                    label = 'Val Error')
        plt.ylim([0,plt_2_top])
        plt.legend()
        plt.show()

    def summarize(self):
        super().raise_override_error()

if __name__ == '__main__':
    m = PriceRegressionModel()
    m.build_model()
    m.run_model(logging_frequency=10)
    m.plot_predictions()
    
