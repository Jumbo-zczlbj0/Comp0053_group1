import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import parse_bpm_lines, closest_frame_idc


"""只取59秒的frames的原因：openface有时候少取几个帧，比如一个60s的视频，应该取60*fps个帧，但是实际取到的少2-3个帧.
   为了方便起见，我们统一取59s."""


class SongEmoDataset(Dataset):
    def __init__(self, root):
        """Each subfolder under 'root' directory contains data collected from one person.
        
          There should be two subfolders 'bpm' & 'openface', and one file 'score.txt' inside
          the subfolder. 'bpm' folder has n txt files storing the bpm values, and 'openface' 
          folderhas n csv files storing the features extracted from videos using OpenFace.
          (n is around 6)
        """
        self.X, self.y = None, None

        for person in os.listdir(root):          
            bpm_dir = os.path.join(root, person, 'bpm')
            opf_dir = os.path.join(root, person, 'openface')   
            n_bpms = len(os.listdir(bpm_dir))
            n_csvs = len(os.listdir(opf_dir))
            with open(os.path.join(root, person, 'score.txt'), 'r') as f:
                scores = f.readlines()
            n_scores = len(scores)
            
            if not (n_bpms == n_csvs and n_bpms == n_scores):
                raise ValueError("Count of csv files, bpm txt files and scores must be the same.")

            fps = 0
            video_fts = None

            for i, file in enumerate(os.listdir(opf_dir)):
                filepath = os.path.join(opf_dir, file)
                df = pd.read_csv(filepath)
                data = df.to_numpy()
                
                # The fps for one person's videos should be the same. Avoid unnecessary assignments.
                if i == 0:
                    fps = round(1 / data[1, 2])
                
                # only take 59 * fps frames, first 5 columns aren't features.
                if i == 0:
                    video_fts = data[np.newaxis, :59 * fps, 5:]
                else:
                    video_fts = np.append(video_fts, data[np.newaxis, :59 * fps, 5:], axis=0)
            
            for j, file in enumerate(os.listdir(bpm_dir)):
                filepath = os.path.join(bpm_dir, file)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                reltime_list, bpm_list = parse_bpm_lines(lines, fps)        
                idc = closest_frame_idc(reltime_list, fps)
                bpms = np.array(bpm_list, dtype=np.int32)
                bpms -= bpms[0]  # Get the relative values to the first bpm.
                
                chosen_idc = idc[::3]
                chosen_bpms = bpms[::3]
                chosen_video_fts = video_fts[j, chosen_idc]
                features = np.concatenate([chosen_video_fts, chosen_bpms.reshape(-1, 1)], axis=1)
                labels = np.zeros(features.shape[0])
                labels.fill(scores[j])  # auto-cast available
                
                if self.X is None:           
                    self.X = features
                    self.y = labels
                else:
                    self.X = np.concatenate([self.X, features], axis=0)
                    self.y = np.concatenate([self.y, labels], axis=0)
                
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    def __len__(self):
        return self.y.shape[0]
