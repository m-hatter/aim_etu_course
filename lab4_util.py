import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model,
          loss_fn,
          optimizer,
          train_data_loader, 
          val_data_loader, *,
          max_epochs,
          device=None,
          early_stopping_metric=None,
          early_stopping_patience=None,
          collect_metrics=None,
          verbose=True):

    def train_epoch():
        total_loss = 0.0
        metric_values = {metric_name: 0.0
                         for metric_name, _ in collect_metrics}
        total_batches = 0
        for X, y in tqdm(train_data_loader) if verbose else train_data_loader:
            optimizer.zero_grad()
            y_hat = model(X.to(device)).reshape_as(y)
            loss = loss_fn(y_hat, y.to(torch.float32).to(device))
            loss.backward()
            optimizer.step()
            # Сбор значений метрик
            y_hat = y_hat.detach().to('cpu')
            for metric_name, metric_foo in collect_metrics:
                metric_values[metric_name] += metric_foo(y_hat, y)
            total_loss += loss.detach().to('cpu').item()
            total_batches += 1
        for metric_name in metric_values:
            metric_values[metric_name] /= total_batches
        metric_values['loss'] = total_loss / total_batches
        return metric_values, total_batches
    
    def evaluate():
        total_loss = 0.0
        metric_values = {metric_name: 0.0
                         for metric_name, _ in collect_metrics}
        total_batches = 0
        with torch.no_grad():
            for X, y in val_data_loader:
                y_hat = model(X.to(device)).reshape_as(y)
                loss = loss_fn(y_hat, y.to(torch.float32).to(device))
                # Сбор значений метрик
                y_hat = y_hat.detach().to('cpu')
                for metric_name, metric_foo in collect_metrics:
                    metric_values[metric_name] += metric_foo(y_hat, y)
                total_loss += loss.detach().to('cpu').item()
                total_batches += 1
        for metric_name in metric_values:
            metric_values[metric_name] /= total_batches
        metric_values['loss'] = total_loss / total_batches
        return metric_values, total_batches

    def update_metrics(x, y):
        for key in x:
            x[key].append(y[key])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    collect_metrics = collect_metrics or []

    m = 1
    if early_stopping_metric is not None:
        if len(early_stopping_metric) > 0 and early_stopping_metric[0] == '-':
            early_stopping_metric = early_stopping_metric[1:]
            m = -1
    
    early_stopping_best_value = m * (-float('inf'))
    early_stopping_best_epoch = None

    early_stopping_patience = early_stopping_patience or 0
    
    # Словарь для хранения истории изменения метрик
    history = {'train': {'loss': []},
               'val': {'loss': []}}
    for metric_type in ['train', 'val']:
        history[metric_type].update({metric_name: [] 
                                    for metric_name, _ in collect_metrics})
    
    model.to(device)
    
    for epoch in range(max_epochs):
        
        model.train()
        metric_values, _ = train_epoch()
        update_metrics(history['train'], metric_values)

        if verbose:
            print(f'Epoch {epoch}: train loss {metric_values["loss"]:.5f}')
        
        model.eval()
        metric_values, _ = evaluate()
        update_metrics(history['val'], metric_values)
        
        if verbose:
            print(f'Epoch {epoch}: val loss {metric_values["loss"]:.5f}')

        if early_stopping_metric is not None:
            if m * history['val'][early_stopping_metric][-1] > m * early_stopping_best_value:
                early_stopping_best_value = history['val'][early_stopping_metric][-1]
                early_stopping_best_epoch = epoch
            else:
                if epoch - early_stopping_best_epoch > early_stopping_patience:
                    break
    
    return history


def prettify(s: str):
    if len(s) > 1:
        return s[0].upper() + s[1:]
    return s

def plot_history(history, param, prettify_labels=True, **kwargs):
    plt.plot(history['train'][param])
    plt.plot(history['val'][param])
    plt.xlabel('Epochs')
    plt.ylabel(prettify(param) if prettify_labels else param)
    plt.legend(['Train', 'Validation'])
    for k, v in kwargs.items():
        getattr(plt, k)(v)    
    plt.show()

def accuracy(model, loader):
    accum = 0
    n = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for x, y in loader:
        x = x.to(device)
        y_hat = (model(x).to('cpu') > 0.5).reshape_as(y)
        accum += (y == y_hat).to(torch.float32).mean().item()
        n += 1
    return accum / n
