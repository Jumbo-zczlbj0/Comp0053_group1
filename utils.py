import os
import pandas as pd
import numpy as np
from datetime import timedelta


def parse_bpm_lines(lines, fps):
    """Convert a list of timestamp and bpm strings to a list of relative times(in milliseconds)
       and bpm values. Video's FPS is needed for computing upper bound of relative times."""
    
    # line example: '13:26:08.293 -> BPM: 101'
    bpm_list = [int(line.split()[-1]) for line in lines]
    timestr_list = [line.split()[0] for line in lines]
    time_list = []
    reltime_list = []
    
    for s in timestr_list:
        # Parse the time string, an example is: '13:26:08:293'.
        hours = int(s[:2])
        minutes = int(s[3:5])
        seconds = int(s[6:8])
        milliseconds = int(s[-3:])
        time_list.append(timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds))
        
    i = 0
    for t in time_list:
        reltime = int((t - time_list[0]).total_seconds() * 1000)
        # Convert float to int because we don't need such accuracy, 
        # change it if you need more accurate values.
        
        if reltime > 1000 * (59 - round(1 / fps, 3)):
            # We only take 59 * fps frames. So relative time ∈ [0, 59 - 1 / fps] seconds.
            # In csv files, relative times are rounded to 3 decimal places, so we make it consistent here.
            break
        reltime_list.append(reltime)
        i += 1
    
    return reltime_list, bpm_list[:i]


def closest_frame_idc(reltime_list, fps):
    """Given a list of relative times(in milliseconds) and video's fps, returns a list of indices,
        which point to the closest frames in video data according to the respective relative times
        in reltime_list"""
    
    times = np.array(reltime_list)
    return np.around(times * fps / 1000).astype(np.int32)


def save_dataset_tensor(root, dest):
    """Get dataset from data and save numpy arrays to dest dir."""
    
    X, y = None, None

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

            if X is None:           
                X = features
                y = labels
            else:
                X = np.concatenate([X, features], axis=0)
                y = np.concatenate([y, labels], axis=0)
    
    path1 = os.path.join(dest, 'X.npy')
    path2 = os.path.join(dest, 'y.npy')
    np.save(path1, X, allow_pickle=False)
    np.save(path2, y, allow_pickle=False)
    print(f"numpy arrays saved at {dest}.")
    
    
def save_bpm_dataset_tensor(root, dest):
    """Get dataset which has only bpm sequence as features, and save numpy arrays to dest dir."""
    
    X, y = None, None

    for i, person in enumerate(os.listdir(root)):     
        # Due to incompeteness of data from the last person, we don't use that data.
        if i == 6:
            break
            
        bpm_dir = os.path.join(root, person, 'bpm')
        opf_dir = os.path.join(root, person, 'openface')   
        with open(os.path.join(root, person, 'score.txt'), 'r') as f:
            scores = f.readlines()
            
        file = os.listdir(opf_dir)[0]
        filepath = os.path.join(opf_dir, file)
        df = pd.read_csv(filepath)
        data = df.to_numpy()
        fps = round(1 / data[1, 2])
        
        for j, file in enumerate(os.listdir(bpm_dir)):
            # These files have too few bpm values, skip this file.
            if i == 4 and (j == 3 or j == 5):
                continue
                
            filepath = os.path.join(bpm_dir, file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            _, bpm_list = parse_bpm_lines(lines, fps)        
            bpms = np.array(bpm_list, dtype=np.int32)
            bpms -= bpms[0]  # Get the relative values to the first bpm.
            
            # People have different numbers of bpms, but we need to ensure the dimensions are the same.
            features = bpms[:60].reshape(1, -1) 
            score = np.array(scores[j], dtype=np.float64).reshape(1)
            
            if X is None:           
                X = features
                y = score
            else:
                X = np.concatenate([X, features], axis=0)
                y = np.concatenate([y, score], axis=0)
                
    path1 = os.path.join(dest, 'X_bpm.npy')
    path2 = os.path.join(dest, 'y_bpm.npy')
    np.save(path1, X, allow_pickle=False)
    np.save(path2, y, allow_pickle=False)
    print(f"numpy arrays saved at {dest}.")
