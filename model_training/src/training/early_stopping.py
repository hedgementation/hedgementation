import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.count = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.count = 0
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}. Resetting early stop counter.")
        else:
            self.count += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.count} out of {self.patience}')

        if self.count >= self.patience:
            self.early_stop = True