import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import h5py
import json



class Emosign_Dataset(Dataset):
    def __init__(self, vid_path):
        self.data_all=pd.read_json('./prefeature/metainfo.json',orient='records',lines=True,dtype={'video_id': str})
        self.vid=[]
        with open(vid_path, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
        self.data.reset_index(inplace=True)
        self.ocr_pattern_fea_path='./prefeature/preprocess_ocr/sam'

        self.ocr_phrase_fea_path='./prefeature/preprocess_ocr/ocr_phrase_fea.pkl'
        with open(self.ocr_phrase_fea_path, 'rb') as f:
            self.ocr_phrase = torch.load(f)

        self.text_semantic_fea_path='./prefeature/preprocess_text/sem_text_fea.pkl'
        with open(self.text_semantic_fea_path, 'rb') as f:
            self.text_semantic_fea = torch.load(f)

        self.text_emo_fea_path='./prefeature/preprocess_text/emo_text_fea.pkl'
        with open(self.text_emo_fea_path, 'rb') as f:
            self.text_emo_fea = torch.load(f)

        self.audio_fea_path='./prefeature/preprocess_audio'
        self.visual_fea_path='./prefeature/preprocess_visual'

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation']=='fake' else 0
        fps=torch.tensor(item['fps'])
        total_frame=torch.tensor(item['frame_count'])
        visual_time_region=torch.tensor(item['transnetv2_segs'])
        label = torch.tensor(label)

        all_phrase_semantic_fea=self.text_semantic_fea['last_hidden_state'][vid]
        all_phrase_emo_fea=self.text_emo_fea['pooler_output'][vid]

        v_fea_path=os.path.join(self.visual_fea_path,vid+'.pkl')
        raw_visual_frames=torch.tensor(torch.load(open(v_fea_path,'rb'))) 

        a_fea_path=os.path.join(self.audio_fea_path,vid+'.pkl')
        raw_audio_emo=torch.load(open(a_fea_path,'rb')) #1*768

        ocr_pattern_fea_file_path=os.path.join(self.ocr_pattern_fea_path,vid,'r0.pkl') 
        ocr_pattern_fea=torch.tensor(torch.load(open(ocr_pattern_fea_file_path,'rb'))) 

        ocr_phrase_fea=self.ocr_phrase['ocr_phrase_fea'][vid] 
        ocr_time_region=self.ocr_phrase['ocr_time_region'][vid] 

        v_fea_path=os.path.join(self.visual_fea_path,vid+'.pkl')
        raw_visual_frames=torch.tensor(torch.load(open(v_fea_path,'rb')))

        return {
            'vid': vid,
            'label': label,
            'fps': fps,
            'total_frame': total_frame,
            'all_phrase_semantic_fea': all_phrase_semantic_fea,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': raw_audio_emo,
            'ocr_pattern_fea': ocr_pattern_fea,
            'ocr_phrase_fea': ocr_phrase_fea,
            'ocr_time_region': ocr_time_region,
            'visual_time_region': visual_time_region
        }
