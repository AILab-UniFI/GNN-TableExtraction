from math import inf
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from src.components.graphs.labels import LableModification
from src.utils.const import Categories_names

from src.utils.paths import CMS

class EarlyStopping:
    """A Class to store the best model scores
    """
    def __init__(self, weights, name, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.weights_path = weights
        dt = datetime.datetime.now()
        self.dt = dt.strftime("%d%m_%H%M")
        self.name = name

    def step(self, loss, model):
        score = loss
        if self.best_score is None:
            self.save_checkpoint(model)
            self.best_score = loss
        elif score > self.best_score:
            self.counter += 1
            # if self.patience != inf: 
                # print(f' -! EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.save_checkpoint(model)
                self.early_stop = True
        else:
            # print(" -> NEW Val Loss: decreased from {} to {}".format(self.best_score, score))
            self.save_checkpoint(model)
            self.best_score = score
            self.counter = 0
        
        return self.early_stop, self.counter

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), f'{self.weights_path}/{self.name}.pt')


def evaluate_test(g, model, labels, idxs):
    model.eval()
    with torch.no_grad():
        logits = model(g, idxs)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), indices

def cm(y_pred, y_true, logs):
    """Print the confusion matrix depending on the task, if binary or not
    """
    classes = ('Text', 'Title', 'List', 'Table-colh', 'Table-colp', 'Table-cell', 'Caption', 'Other', 'Image')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(classes))], normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    dt = str(datetime.datetime.now())
    plt.savefig(CMS / logs + '.png')
    return

def new_cm(y_pred, y_true, logs, label_converted:LableModification=None, converted=False):
    """Print the confusion matrix depending on the task, if binary or not
    """
    classes = [ cn.name for cn in Categories_names ]
    labels = [ cn.value for cn in Categories_names ]

    if label_converted == None: raise ValueError('ERROR: label_converted is None')

    print(f'ATTENTION: classes and labels conversion')
    labels = [i for i in list(map(lambda x: label_converted.origin_to_conv[x], labels)) if i != None]
    classes = [Categories_names(label_converted.conv_to_origin[lab]).name for lab in labels]
    
    if not converted:
        y_true = list(map(lambda x: label_converted.origin_to_conv[x], y_true))
        y_pred = list(map(lambda x: label_converted.origin_to_conv[x], y_pred))
        
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    dt = str(datetime.datetime.now())
    plt.savefig(CMS / f'{logs}.png')
    return