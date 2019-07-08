from tensorflow import keras
import sys
import itertools

class PeriodicEpochLogger(keras.callbacks.ProgbarLogger):
    def __init__(self, frequency, logging_params=[]):
        super().__init__()
        self.episode = frequency
        self.logging_params = logging_params

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['samples']
            self.target = target
            self.progbar = keras.utils.Progbar(target=self.target, verbose=self.verbose,
                                   stateful_metrics=self.stateful_metrics)
        self.seen = 0
 
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs):

        def interleave_log_dictionary():
            logging_values = [logs[param_name] for param_name in self.logging_params]
            return list(itertools.chain(*zip(self.logging_params, logging_values)))
        
        super().on_batch_end(epoch, logs)
        if epoch % self.episode == self.episode - 1:
              self.progbar.update(self.seen, self.log_values)
              print("Epoch {} - {}={:.4}, {}={:.4}, {}={:.4}, {}={:.4}"
                      .format(epoch + 1, *(interleave_log_dictionary())))
            
