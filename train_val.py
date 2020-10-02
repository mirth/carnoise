import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from tqdm import tqdm
import baker
from vggish_input import wavfile_to_examples
from hubconf import vggish
from vggish_params import WINDOW_MULTIPLIER


class NoiseDataSet(Dataset):
    def __init__(self, X, y):
        super().__init__()
        assert(len(X) == len(y))

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        filename = self.X[idx]
        wav = wavfile_to_examples(filename)
        wav = wav.cuda()

        target = self.y[idx]

        return wav, target

class NoiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vggish(
            pretrained=False,
            preprocess = False,
            postprocess = False,
        )
        self.vgg.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6 * WINDOW_MULTIPLIER, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.vgg(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.squeeze()

        return x

def squeeze_collate(batch):
    batch_x, batch_y = list(zip(*batch))
    batch_y = np.array(batch_y, dtype=np.float32)
    batch_y = torch.from_numpy(batch_y)
    batch_x = torch.cat(batch_x, dim=0)

    return batch_x, batch_y

def get_data_loaders(batch_size):
    df = pd.read_csv('/home/tolik/data/carnoises/noise_and_healthy_fixed3.csv')#, nrows=100)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.sample_uri.values
    y = df.target.values
    groups = df.ytid.values

    gss = GroupShuffleSplit(n_splits=1,  test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    train_dataset = NoiseDataSet(X_train, y_train)
    val_dataset = NoiseDataSet(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=squeeze_collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=squeeze_collate, drop_last=True)

    return train_loader, val_loader

def round_output(x):
    y_pred, y_true = x
    y_pred = torch.round(y_pred)

    return y_pred, y_true

@baker.command
def run(batch_size=8, epochs=5, device='cuda'):
    train_loader, val_loader = get_data_loaders(batch_size)

    model = NoiseClassifier()
    model = model.to(device)

    model.eval()

    optimizer = Adam(model.parameters())
    criterion = nn.BCELoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    precision = Precision(average=False, output_transform=round_output)
    recall = Recall(average=False, output_transform=round_output)
    F1 = (precision * recall * 2 / (precision + recall)).mean()
    val_metrics = {
        'accuracy': Accuracy(output_transform=round_output),
        'loss': Loss(criterion),
        'precision': precision,
        'recall': recall,
        'F1': F1,
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    desc = 'ITERATION - loss: {:.2f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    log_interval = 1
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics

        tqdm.write(
            'Training Results - Epoch: {} Precision {:.2f}, Recall: {:.2f}, F1: {:.2f}, Avg loss: {:.2f}'.format(
                engine.state.epoch,
                metrics['precision'],
                metrics['recall'],
                metrics['F1'],
                metrics['loss'],
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        tqdm.write(
            'Validation Results - Epoch: {} Precision {:.2f}, Recall: {:.2f}, F1: {:.2f}, Avg loss: {:.2f}'.format(
                engine.state.epoch,
                metrics['precision'],
                metrics['recall'],
                metrics['F1'],
                metrics['loss'],
            )
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()

if __name__ == '__main__':
    baker.run()

