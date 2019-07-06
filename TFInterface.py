# MODULE
from abc import ABC, abstractmethod

class AbstractClassifier(ABC):

    def __init__(self):
        super().__init__() 

    def raise_override_error(self):
        raise TypeError('Abstract method ' + self._class.__name__                                           + '.' + self._function + 'must be overridden')
    @abstractmethod
    def preprocess(self, **kwargs):
        self.raise_override_error()

    @abstractmethod
    def train(self, **kwargs):
        self.raise_override_error()

    @abstractmethod
    def eval(self, **kwargs):
        self.raise_override_error()

    @abstractmethod
    def build_model(self, **kwargs):
        self.raise_override_error()

    @abstractmethod
    def run_model(self, **kwargs):
        self.train()
        self.eval()

    @abstractmethod    
    def model_check(self):
        if self.model is None:
            print("No model compiled!")
            sys.exit(0)

    @abstractmethod
    def plot_predictions(self, **kwargs):
        self.raise_override_error()

    @abstractmethod
    def preview(self):
        self.raise_override_error()

    @abstractmethod
    def summarize(self):
        self.model.summary()


