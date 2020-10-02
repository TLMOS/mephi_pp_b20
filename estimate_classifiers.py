import configparser
import sys
import cv2
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import time
import json
from timm.models.efficientnet import tf_efficientnet_b0_ns, tf_efficientnet_b1_ns, \
tf_efficientnet_b2_ns, tf_efficientnet_b3_ns, tf_efficientnet_b4_ns, tf_efficientnet_b5_ns, \
tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, tf_efficientnet_b7_ns
import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from sklearn.metrics import accuracy_score
from ptflops.flops_counter import get_model_complexity_info
from functools import partial
from log import Log
from deep_fake_classifier_dataset import DeepFakeClassifierDataset

encoder_inits = {
    'efficientnet_b0':partial(tf_efficientnet_b0_ns, pretrained=True),
    'efficientnet_b1':partial(tf_efficientnet_b1_ns, pretrained=True),
    'efficientnet_b2':partial(tf_efficientnet_b2_ns, pretrained=True),
    'efficientnet_b3':partial(tf_efficientnet_b3_ns, pretrained=True),
    'efficientnet_b4':partial(tf_efficientnet_b4_ns, pretrained=True),
    'efficientnet_b5':partial(tf_efficientnet_b5_ns, pretrained=True),
    'efficientnet_b6':partial(tf_efficientnet_b6_ns, pretrained=True),
    'efficientnet_b7':partial(tf_efficientnet_b7_ns, pretrained=True)
}

input_resolutions = {
      'efficientnet_b0': 224,
      'efficientnet_b1': 240,
      'efficientnet_b2': 260,
      'efficientnet_b3': 300,
      'efficientnet_b4': 380,
      'efficientnet_b5': 456,
      'efficientnet_b6': 528,
      'efficientnet_b7': 600
}

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, encoder_features, dropout_rate=0.0):
        super().__init__()
        self.encoder = encoder
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_features, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
    def freeze_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self):
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
            
def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

def train_epoch(model, optimizer, loss_functions, train_data_loader, history):
    for imgs, labels, *_ in train_data_loader:
        imgs = imgs.type(torch.float).cuda()
        labels = labels.type(torch.float).reshape(-1, 1).cuda()
        optimizer.zero_grad()
        label_predicted = model(imgs)
        loss = loss_functions(label_predicted, labels)
        loss.backward()
        optimizer.step()
        history.append(np.mean(loss.cpu().detach().numpy()))
        
def evaluate_classifier(model, data, metric, average_prediction=None, log=None):
    model.eval()
    
    predicted_frame_labels = {}
    labels = {}
    for vid in data.metadata['vid'].unique():
        predicted_frame_labels[vid] = []
        labels[vid] = int(data.metadata[data.metadata['vid'] == vid]['label'].iloc[0])

    data_loader = DataLoader(data)
    iterable = log.tqdm(data_loader, 'Evaluating') if log is not None else tqdm(data_loader)
    for img, label, vid, cid in iterable:
        img = img.type(torch.float).cuda()
        predicted_label = model(img).cpu().detach().numpy()
        predicted_frame_labels[vid[0]].append(predicted_label)
    input_size = img.cpu().detach().numpy().shape[1:]
    
    predicted_labels = {}
    if average_prediction is not None:
        for vid in predicted_frame_labels:
            predicted_labels[vid] = int(average_prediction(predicted_frame_labels[vid]) >= 0.5)
    else:
        for vid in predicted_frame_labels:
            predicted_labels[vid] = int(np.mean(predicted_frame_labels[vid]) >= 0.5)
            
    score = metric(list(labels.values()), list(predicted_labels.values()))
    
    if getattr(model, 'compute_average_flops_cost', None) is not None:
        macs, params = model.compute_average_flops_cost()
    else:
        macs, params = get_model_complexity_info(model, input_size,
                                                 print_per_layer_stat=False,
                                                 as_strings=False)
    
    return score, macs, params, labels, predicted_labels

def estimate_classifiers(data_train, data_test, classifier, classifier_params, encoders,
                         loss_function, metric, batch_size, epochs, freeze_epochs=None,
                         average_prediction=None, save_models=False):
    log = Log()
    log.log('{:-^70}'.format('Estimating {} classifiers'.format(len(encoders))))
    summary = []
    for encoder_id in encoders:
        log.log('{:-^70}'.format(encoder_id))
        data_train.set_image_size(input_resolutions[encoder_id])
        data_test.set_image_size(input_resolutions[encoder_id])
        encoder = encoder_inits[encoder_id]().cuda()
        encoder_features = encoder.num_features
        model = classifier(encoder, encoder_features, *classifier_params).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        loss_function = loss_function.cuda()
        log.log('Initialized')
        
        history = []
        mean_dur = 0
        model.train()
        if freeze_epochs is not None and freeze_epochs != 0:
            model.freeze_encoder()

        for epoch in log.tqdm(range(epochs), 'Training'):
            if epoch == freeze_epochs:
                model.unfreeze_encoder()

            train_data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                                           drop_last=True)

            train_epoch(model, optimizer, loss_function, train_data_loader, history)

        log.log('Loss: {}'.format(history[-1]))
        
        score, macs, params, labels, predicted_labels = evaluate_classifier(model, data_test, metric,
                                                  average_prediction=average_prediction,
                                                  log=log)
        log.log("Score:     {}\nMACs(B):   {}\nParams(M): {}".format(
            round(score, 6), round(macs/10**9, 3), round(params/10**6, 3)))
        
        summary.append({'encoder':encoder_id, 'score':float(score), 'macs':int(macs),
                        'params':int(params), 'history':list(map(float, history))})
        
        save_time = int(time.time())
        prediction = pd.DataFrame({'vid':list(predicted_labels.keys()), 'label':list(labels.values()), 
                           'prediction':list(predicted_labels.values())})
        prediction.to_csv(os.path.join(predictions_path,
                                       "{}_{}_prediction.csv".format(save_time, encoder_id)),
                         index=False)
        log.log('Prediction saved')
        
        if save_models:
            torch.save(model, os.path.join(models_path,
                                           "{}_{}_model.pth".format(save_time, encoder_id)))
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(models_path,
                                    "{}_{}_state_dict.pth".format(save_time, encoder_id)))
            log.log('Model saved')
    log.log('{:-^70}'.format('Complete'))
    return summary



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    data_path           = config['Paths']['data_path']
    processed_data_path = config['Paths']['processed_data_path']
    models_path         = config['Paths']['models_path']
    predictions_path    = config['Paths']['predictions_path']
    summary_path        = config['Paths']['summary_path']
    encoders            = config['Encoders']['encoders'].split(',')
    batch_size          = int(config['Training']['batch_size'])
    epochs              = int(config['Training']['epochs'])
    freeze_epochs       = int(config['Training']['freeze_epochs'])
    dropout_rate        = float(config['Training']['dropout_rate'])
    learning_rate       = float(config['Training']['learning_rate'])
    
    if not os.path.isdir(processed_data_path):
        os.makedirs(processed_data_path)
        os.makedirs(os.path.join(processed_data_path, 'crops'))
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    if not os.path.isdir(predictions_path):
        os.makedirs(predictions_path)
    if not os.path.isdir(summary_path):
        os.makedirs(summary_path)
    
    metadata_train = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    metadata_test = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))
    data_train = DeepFakeClassifierDataset(metadata_train, processed_data_path, mode='train')
    data_test = DeepFakeClassifierDataset(metadata_test, processed_data_path, mode='test')
    
    summary = estimate_classifiers(data_train, data_test, classifier=DeepFakeClassifier, 
                     classifier_params=(dropout_rate,), encoders=encoders, 
                     loss_function=torch.nn.BCELoss(), metric=accuracy_score,
                     batch_size=batch_size, epochs=epochs, freeze_epochs=freeze_epochs,
                     average_prediction=confident_strategy, save_models=True)
    
    save_time = int(time.time())
    with open(os.path.join(summary_path, '{}_summary.json'.format(save_time)), 'w') as f:
        json.dump(summary, f)