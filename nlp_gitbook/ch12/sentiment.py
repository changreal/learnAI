
class Sentiment(object):
    def __init__(self):
        self.classifier = Bayes()

    def save(self, fname, iszip=True):
        pass

    def load(self, fname=data_path, iszip=True):
        pass

    def handle(self):
        pass

    def train(self):
        pass

    def classify(self, sent):
        pass

