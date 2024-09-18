import torch
from .transformer import *
import pandas as pd
import json
from .attention import *




class Textual(torch.nn.Module):
    def __init__(self):
        super(Textual,self).__init__()
        self.mlp_text_emo = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1))
        self.trm_emo = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)

    def forward(self, **kwargs):
        all_phrase_emo_fea = kwargs['all_phrase_emo_fea']
        raw_t_fea_emo = self.mlp_text_emo(all_phrase_emo_fea).unsqueeze(1)
        return raw_t_fea_emo

class Audio(torch.nn.Module):
    def __init__(self):
        super(Audio, self).__init__()
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.ReLU(), nn.Dropout(0.1))

    def forward(self, **kwargs):
        raw_audio_emo = kwargs['raw_audio_emo']
        raw_a_fea_emo = self.mlp_audio(raw_audio_emo).unsqueeze(1)
        return raw_a_fea_emo

class Video(torch.nn.Module):
    def __init__(self):
        super(Video, self).__init__()
        self.input_visual_frames = 83
        self.encoded_text_semantic_fea_dim = 512
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(),
                                               nn.Dropout(0.1))
        self.mlp_img = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1))
        self.co_attention_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                            visual_len=self.input_visual_frames, sen_len=512, fea_v=128, fea_s=128,
                                            pos=False)
        self.trm_semantic = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)

    def forward(self, **kwargs):
        all_phrase_semantic_fea = kwargs['all_phrase_semantic_fea']
        raw_visual_frames = kwargs['raw_visual_frames']
        raw_t_fea_semantic = self.mlp_text_semantic(all_phrase_semantic_fea)
        raw_v_fea = self.mlp_img(raw_visual_frames)
        content_v, content_t = self.co_attention_tv(v=raw_v_fea, s=raw_t_fea_semantic, v_len=raw_v_fea.shape[1],
                                                        s_len=raw_t_fea_semantic.shape[1])
        content_v = torch.mean(content_v, -2)
        content_t = torch.mean(content_t, -2)
        fusion_semantic_fea = self.trm_semantic(torch.cat((content_t.unsqueeze(1), content_v.unsqueeze(1)), 1))
        fusion_semantic_fea = torch.mean(fusion_semantic_fea, 1)
        return fusion_semantic_fea

class Emosign_Model(torch.nn.Module):
    def __init__(self):
        super(Emosign_Model, self).__init__()
        self.text_branch = Textual()
        self.audio_branch = Audio()

        self.video_branch = Video()
        self.trm_emo = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.content_classifier = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))


    def forward(self, **kwargs):
        output_text = self.text_branch(**kwargs)
        output_audio = self.audio_branch(**kwargs)
        output_video = self.video_branch(**kwargs)
        fusion_emo_fea = self.trm_emo(torch.cat((output_text, output_audio), 1))
        fusion_emo_fea = torch.mean(fusion_emo_fea, 1)
        msam_fea = torch.cat((fusion_emo_fea, output_video), 1)
        output = self.content_classifier(msam_fea)
        return output