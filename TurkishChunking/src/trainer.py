import sklearn_crfsuite
from src.features import sent2features, sent2labels, load_conll
import os
import joblib

class ChunkingTrainer:
    def __init__(self):
        self.models = {}
        self.targets = ['OUTER', 'INNER', 'CLAUSE']

    def train(self, train_file):
        sentences = load_conll(train_file)
        X_train = [sent2features(s) for s in sentences]
        
        for idx, target in enumerate(self.targets):
            print(f"Training model for {target}...")
            y_train = [sent2labels(s, idx + 1) for s in sentences]
            
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            crf.fit(X_train, y_train)
            self.models[target] = crf
            
    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for target, model in self.models.items():
            joblib.dump(model, os.path.join(path, f"{target}_model.pkl"))
            
    def load_models(self, path):
        for target in self.targets:
            self.models[target] = joblib.load(os.path.join(path, f"{target}_model.pkl"))

    def predict(self, X):
        results = {}
        for target, model in self.models.items():
            results[target] = model.predict(X)
        return results
