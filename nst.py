from keras.applications import vgg19
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as K
import numpy as np
import NSTUtils
import sys
import time
import PIL
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

class StyleTransferer():
    def __init__(self, img_in_content, img_in_style, outpath=None, img_rows=512,
            content_weight=0.01, style_weight=0.75, tv_weight=1.0):
        self.iterations = 0
        self.content_image_path = img_in_content
        self.style_image_path = img_in_style
        if outpath is not None:
            self.out = outpath
        else:
            self.out = str(time.time())

        (true_width, true_height) = load_img(img_in_content).size
        self.nrows = img_rows
        self.ncols = int(true_width * self.nrows / true_height)

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        self.content_tensor = K.variable(self.preprocess_image(img_in_content))
        self.style_tensor = K.variable(self.preprocess_image(img_in_style))
        if K.image_data_format() == 'channels_first':
            self.combination_image = K.placeholder((1, 3, self.nrows, self.ncols))
        else:
            self.combination_image = K.placeholder((1, self.nrows, self.ncols, 3))
        self.input_tensor = K.concatenate([self.content_tensor, self.style_tensor, 
            self.combination_image], axis=0)

        self.history = []
                

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.nrows, self.ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def build_model(self):
        self.model = vgg19.VGG19(input_tensor=self.input_tensor,
                    weights='imagenet', include_top=False)
        print("Model loaded.")
        layer_features = NSTUtils.getVGG19Layer('block5_conv2', self.model)
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = K.variable(0.0)
        content_loss = self.content_weight * NSTUtils.content_loss(base_image_features,
                                      combination_features)
        loss += content_loss
        feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1', 
                  'block5_conv1']
        fmap_size = self.nrows * self.ncols
        style_loss = 0
        for layer_name in feature_layers:
            layer_features = NSTUtils.getVGG19Layer(layer_name, self.model)
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = NSTUtils.style_loss(style_reference_features, combination_features, fmap_size)
            style_loss += (self.style_weight / len(feature_layers)) * sl
        loss += style_loss
        tv_loss = self.tv_weight * NSTUtils.total_variation_loss(self.combination_image, self.nrows, self.ncols)
        loss += tv_loss
        grads = K.gradients(loss, self.combination_image)

        outputs = [loss, content_loss, style_loss, tv_loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)
        self.get_loss_and_grads = K.function([self.combination_image], outputs)

    def single_iteration(self, image_tensor):
        sys.stdout.write("Iteration {}: ".format(self.iterations))
        sys.stdout.flush()
        self.iterations += 1
        start_time = time.time()
        if K.image_data_format() == 'channels_first':
            image_tensor = image_tensor.reshape((1, 3, self.nrows, self.ncols))
        else:
            image_tensor = image_tensor.reshape((1, self.nrows, self.ncols, 3))
        outs = self.get_loss_and_grads([image_tensor])
        loss_values = outs[0:4]
        self.history.append(loss_values)
        sys.stdout.write("loss: {:<4.4g} - content: {:<4.4g} - style: {:<4.4g} - tv: {:<4.4g}".format(loss_values[0], loss_values[1], loss_values[2], loss_values[3]))
        sys.stdout.flush()
        if len(outs[4:]) == 1:
            grad_values = outs[4].flatten().astype('float64')
        else:
            grad_values = np.array(outs[4:]).flatten().astype('float64')
        self.loss_cache = loss_values[0]
        self.grad_cache = grad_values
        end_time = time.time()
        sys.stdout.write(' - took {:5.3f}s.\n'.format(end_time - start_time))
        return loss_values, grad_values

    def loss(self, x):
        loss_value, grad_values = self.single_iteration(x)
        self.loss_value = loss_value[0]
        self.grad_values = grad_values
        return self.loss_cache

    def grad(self, _):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def train_model(self, n_iters=10, save_every=50):
        x = self.preprocess_image(self.content_image_path)    
        for i in range(n_iters):
            sys.stdout.flush()
            start_time = time.time()

            x, min_val, info = fmin_l_bfgs_b(self.loss, x.flatten(),
                           fprime=self.grad, maxfun=50)
            end_time = time.time()
            sys.stdout.flush()
            if (i + 1) % save_every == 0:
                 # save current generated image
                img = NSTUtils.tensor_to_image(x.copy(), self.nrows, self.ncols)
                fname = "nst-output/nst_{}_after_{}.png".format(self.out, i+1)
                save_img(fname, img)
                print('Image saved as', fname)

    def plot_loss(self):
        print("Plotting...")
        start_time = time.time()
        indices = range(len(self.history))
        hist_types = zip(*self.history)
        end_time = time.time()
        print("Zipped history (took {:.4}s).".format(end_time-start_time))
        plt.plot(indices, hist_types[0], color='m', linestyle='-', label='Loss')
        plt.plot(indices, hist_types[1], color='r', linestyle='--', label='Content Loss')
        plt.plot(indices, hist_types[2], color='g', linestyle='--', label='Style Loss')
        plt.plot(indices, hist_types[3], color='b', linestyle='--', label='Total Variational Loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    assert(sys.argv[1][-4:].endswith(".jpg"))
    assert(sys.argv[2][-4:].endswith(".jpg"))
    nst = StyleTransferer(sys.argv[1], sys.argv[2])
    nst.build_model()
    nst.train_model(save_every=1)
    nst.plot_loss()
