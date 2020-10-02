from facenet_pytorch.models.mtcnn import MTCNN
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import math
import json
import pandas as pd
from log import Log
import configparser
import sys

def get_scale_coef(w):
    if w <= 300:
        scale_coef = 2
    elif w <= 1000:
        scale_coef = 1
    elif w <= 1900:
        scale_coef = 0.5
    else:
        scale_coef = 0.33
    return scale_coef

class VideoDataset(Dataset):
    def __init__(self, data_path, step=1):
        self.data_path = data_path
        self.step = step
        self.split_types = ['test', 'train']
        
        data = []
        for split in self.split_types:
            for path in os.listdir(os.path.join(data_path, split)):
                if os.path.isdir(os.path.join(data_path, split, path)):
                    for name in os.listdir(os.path.join(data_path, split, path)):
                        if name != 'metadata.json':
                            data.append({'name':name, 'path':path, 'split':split})
        data = pd.DataFrame(data)
        self.data = data
    
    def __getitem__(self, index):
        name, path, split = self.data.loc[index]
        capture = cv2.VideoCapture(os.path.join(self.data_path, split, path, name))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale_coef = get_scale_coef(frame_w)
        vid = os.path.splitext(name)[0]
        
        frames = []
        scaled_frames = []
        for i in range(frame_count):
            capture.grab()
            success, frame = capture.retrieve()
            if success and i%self.step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                scaled_frame = cv2.resize(frame, tuple([int(s*scale_coef) for s in frame.shape[1::-1]]))
                frames.append(frame)
                scaled_frames.append(scaled_frame)
        return vid, frames, scaled_frames, scale_coef
        
    def __len__(self):
        return len(self.data)
    
class FaceDetector():
    def __init__(self, batch_size, thresholds, device=None):
        self.batch_size = batch_size
        self.detector = MTCNN(margin=0, thresholds=thresholds, device=device)
    
    def detect_faces(self, frames, scale_coef):
        boxes = []
        for i in range(math.ceil(len(frames) / self.batch_size)):
            batch_boxes, *_ = self.detector.detect(frames[i*self.batch_size:(i + 1)*self.batch_size])
            boxes += [(b/scale_coef).astype(int).tolist() if b is not None else None for b in batch_boxes]
        return boxes
    
def crop_frame(frame, box, padding_coef):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    w_p = int(w * padding_coef)
    h_p = int(h * padding_coef)
    crop = frame[max(0, y_min - h_p):y_max + h_p, max(0, x_min - w_p):x_max + w_p]
    return crop

def process_videos(dataset, processed_data_path, face_detector, log, padding_coef=0.3):
    loader = DataLoader(dataset, collate_fn=lambda x: x)
    crops_dir = os.path.join(processed_data_path, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    for item in log.tqdm(loader, 'Processing videos'):
        vid, frames, scaled_frames, scale_coef = item[0]
        boxes = face_detector.detect_faces(scaled_frames, scale_coef)
        
        out_dir = os.path.join(crops_dir, vid)
        os.makedirs(out_dir, exist_ok=True) 
        for i in range(len(frames)):
            box = boxes[i]
            if box is not None:
                frame = frames[i]
                crop = crop_frame(frame, box[0], padding_coef)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, "{}.png".format(i)), crop)
                
def save_crop_metadata(data_path, processed_data_path, split):
    video_metadata = {}
    if split == 'train':
        for path in os.listdir(os.path.join(data_path, split)):
            if os.path.isdir(os.path.join(data_path, split, path)):
                with open(os.path.join(data_path, split, path, 'metadata.json'), 'r') as f:
                    metadata_part = json.load(f)
                    video_metadata.update(metadata_part)
    elif split == 'test':
        with open(os.path.join(data_path, split, 'metadata.json'), 'r') as f:
            metadata_part = json.load(f)
            video_metadata.update(metadata_part)
    else:
        return
    
    crops_metadata = []
    for video_name in video_metadata:
        label, *_ = video_metadata[video_name].values()
        label = True if label == 'REAL' else False
        vid = video_name.split('.')[0]
        crops_path = os.path.join(processed_data_path, 'crops', vid)
        for crop_name in os.listdir(crops_path):
            cid = os.path.splitext(crop_name)[0]
            crops_metadata.append({'vid':vid, 'cid':cid, 'label':label})
    crops_metadata = pd.DataFrame(crops_metadata)
    crops_metadata.to_csv(os.path.join(processed_data_path, '{}.csv'.format(split)),
                          index=False)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    data_path           = config['Paths']['data_path']
    processed_data_path = config['Paths']['processed_data_path']
    padding_coef        = float(config['Preprocessing']['padding_coef'])
    frame_step          = int(config['Preprocessing']['frame_step'])
    detector_batch_size = int(config['Preprocessing']['detector_batch_size'])
    thresholds          = list(map(float, config['Preprocessing']['thresholds'].split(',')))
    
    if not os.path.isdir(processed_data_path):
        os.makedirs(processed_data_path)
        os.makedirs(os.path.join(processed_data_path, 'crops'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log = Log()
    dataset = VideoDataset(data_path, step=frame_step)
    log.log('{:-^70}'.format('Processing data'))
    log.log('Video dataset loaded')
    face_detector = FaceDetector(detector_batch_size, thresholds=thresholds, device=device)
    log.log('Face detector initialized')
    process_videos(dataset, processed_data_path, face_detector, log, padding_coef=padding_coef)
    save_crop_metadata(data_path, processed_data_path, 'train')
    save_crop_metadata(data_path, processed_data_path, 'test')
    log.log('Metadata saved')
    log.log('{:-^70}'.format('Complete'))