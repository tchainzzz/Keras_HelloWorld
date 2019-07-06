import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
import numpy as np

class NNClassifier():
    def __init__(self, dataset=keras.datasets.fashion_mnist):
        # dataset
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset.load_data()
        self.model = None

    def plot_image(self, image_index):
        if image_index < len(self.train_images):
            # plot image
            plt.figure()
            plt.imshow(self.train_images[image_index])
            plt.colorbar()
            plt.grid(False)
            plt.show()

    def preprocess(self, preview=False):
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def build2LayerNN(self, layer1_size=128, layer2_size=10, layer1_act=tf.nn.relu, layer2_act=tf.nn.softmax):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.train_images.shape[1], self.train_images.shape[2])),
            keras.layers.Dense(layer1_size, activation=layer1_act),
            keras.layers.Dense(layer2_size, activation=layer2_act)
        ])
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def model_check(self):
        if self.model is None:
            print("No model compiled!")
            sys.exit(0)

    def run_model(self, n_epochs=5):
        self.train(n_epochs)
        self.eval()
            
    def train(self, n_epochs=5):
        self.model_check()
        self.model.fit(self.train_images, self.train_labels, epochs=n_epochs)

    def eval(self):
        self.model_check()
        train_loss, train_acc = self.model.evaluate(self.train_images, self.train_labels)
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print('Train accuracy:', train_acc)
        print('Test accuracy:', test_acc)
        self.plot_predictions()

    def plot_predictions(self):
         class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
         predictions = self.model.predict(self.test_images)

         def plot_image(i, predictions_array, true_label, img):
             predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
             plt.grid(False)
             plt.xticks([])
             plt.yticks([])

             plt.imshow(img, cmap=plt.cm.binary)

             predicted_label = np.argmax(predictions_array)
             if predicted_label == true_label:
                 color = 'blue'
             else:
                color = 'red'
 
             plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

         def plot_value_array(i, predictions_array, true_label):
             predictions_array, true_label = predictions_array[i], true_label[i]
             plt.grid(False)
             plt.xticks([])
             plt.yticks([])
             thisplot = plt.bar(range(10), predictions_array, color="#777777")
             plt.ylim([0, 1])
             predicted_label = np.argmax(predictions_array)

             thisplot[predicted_label].set_color('red')
             thisplot[true_label].set_color('blue')

         num_rows = 5
         num_cols = 3
         num_images = num_rows*num_cols
         plt.figure(figsize=(2*2*num_cols, 2*num_rows))
         for i in range(num_images):
             plt.subplot(num_rows, 2*num_cols, 2*i+1)
             plot_image(i, predictions, self.test_labels, self.test_images)
             plt.subplot(num_rows, 2*num_cols, 2*i+2)
             plot_value_array(i, predictions, self.test_labels)
         plt.show()
 

    def preview(self):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument('--plot', action='store_true', help="Plotting one image")
    psr.add_argument('--index', '-i', type=int, help="Image index to be plotted")
    psr.add_argument('--preview', action='store_true', help="Preview during preprocessing")
    args = psr.parse_args()
    n = NNClassifier()
    if args.plot and args.index:
        n.plot_image(args.index)
    n.preprocess(preview=args.preview)
    n.build2LayerNN()
    n.run_model()


