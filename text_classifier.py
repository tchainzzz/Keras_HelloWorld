import tensorflow as tf
from tensorflow import keras
import argparse
import numpy as np
from TFInterface import AbstractClassifier

class TextClassifier(AbstractClassifier):
    def __init__(self, dataset=keras.datasets.imdb, keep_top=10000):
        self.data = dataset
        self.retained = keep_top
        self.model = None
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = self.data.load_data(num_words=keep_top)
        word_index = dataset.get_word_index()

        # The first indices are reserved
        self.word_index = {k:(v+3) for k,v in word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2  # unknown
        self.word_index["<UNUSED>"] = 3
        self.reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def get_single_word(self, index):
        return reverse_word_index.get(i, '?')

    def decode_review(self, text):
        return ' '.join([self.get_word(i) for i in text])

    def preprocess(self):
        self.train_data = keras.preprocessing.sequence.pad_sequences(self.train_data,
                                                        value=self.word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

        self.test_data = keras.preprocessing.sequence.pad_sequences(self.test_data,
                                                       value=self.word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

    def train(self, train_set_prop=0.8, rand_seed=42, n_epochs=40, batch_size=512, verbose=1):
        super().model_check()
        np.random.seed(rand_seed)
        train_data, dev_data = np.split(np.random.permutation(self.train_data), [int(train_set_prop * len(self.train_data))])
        train_labels, dev_labels = np.split(np.random.permutation(self.train_labels), [int(train_set_prop * len(self.train_labels))])
        history = self.model.fit(train_data,
                    train_labels,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(dev_data, dev_labels),
                    verbose=int(verbose))
        

    def eval(self):
        super().model_check()
        results = self.model.evaluate(self.test_data, self.test_labels)
        print(results)

    def build_model(self, vocab_size=None):
        if vocab_size == None:
            vocab_size = self.retained

        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, 16))
        self.model.add(keras.layers.GlobalAveragePooling1D())
        self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    def run_model(self):
        super().run_model() 

    def summarize(self):
        super().summarize()
    
    def plot_predictions(self):
        pass

    def preview(self):
        pass

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    t = TextClassifier()
    t.preprocess()
    t.build_model()
    t.run_model()

