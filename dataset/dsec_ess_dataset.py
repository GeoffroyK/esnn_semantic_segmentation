import torch
import torch.nn as nn
import torch.nn.functional as F
from eventslicer import EventSlicer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import h5py
import hdf5plugin
import math


class ESS_DSEC_Dataset(Dataset):
    """
    Custom dataset for the DSEC dataset.
    """
    def __init__(self, data_dir, event_dir, num_labels=11, camera='right'):
        self.data_dir = data_dir
        self.event_dir = event_dir

        self.data = []
        self.num_labels = num_labels
        self.timestamps = []
        self.camera = camera
        self.labels = self.get_all_semantic_paths_labels(data_dir)

    def get_event_frames(self, path, timestamp):
        """
        Load event frames from a given path.
        """
        print(timestamp)
        with h5py.File(path, 'r') as f:
            t = f['events']['t']
            x = f['events']['x']
            y = f['events']['y']
            p = f['events']['p']
            offset = int(f['t_offset'][()])
            ms_to_idx = f['ms_to_idx'][()]

            timestamp -= offset

            start_timestamp = timestamp - 5
            end_timestamp = timestamp + 5
            start_idx = ms_to_idx[start_timestamp]
            end_idx = ms_to_idx[end_timestamp]
                
            #idx = ms_to_idx[timestamp]
            #idx = int(idx)
            events = {
                't': t[start_idx:end_idx] + offset,
                'x': x[start_idx:end_idx],
                'y': y[start_idx:end_idx],
                'p': p[start_idx:end_idx]
            }
        return events

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        #sample = self.data[idx]
        #label = self.labels[idx]
        #events = self.get_event_frames(sample, self.timestamps[idx])
        event_path = self.labels[idx]
        event = event_path.split('/') 
        folder = self.event_dir + event[-3] + '/events/' + self.camera + '/' 'events.h5'
        #events = self.get_event_frames(folder, self.timestamps[idx])

        print("Timestamp", self.timestamps[idx])
        print(folder)

        with h5py.File(folder, 'r') as f:
            eventslicer = EventSlicer(f)
            if idx == 0:
                events = eventslicer.get_events(self.timestamps[idx], self.timestamps[idx]+20000)
            else:
                events = eventslicer.get_events(self.timestamps[idx]-10000, self.timestamps[idx]+10000)
        
        return events, self.labels[idx]
            
    def get_all_semantic_paths_labels(self, root_dir):
        semantic_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            filenames.sort()
            if str(self.num_labels) + 'classes' in dirpath:
                for filename in filenames:
                    if filename.endswith('.png'):
                        full_path = os.path.join(dirpath, filename)
                        semantic_paths.append(full_path)
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if 'timestamp' in filename:
                    timestamps_path = os.path.join(dirpath, filename)
                    with open(timestamps_path, 'r') as f:
                        lines = [int(line.strip()) for line in f.readlines()]
                        self.timestamps.extend(lines)

        print(f"Found {len(semantic_paths)} semantic paths.")
        print(f"Found {len(self.timestamps)} timestamps.")
        return semantic_paths

dt = ESS_DSEC_Dataset(data_dir='/media/geoffroy/T7/Thèse/Datasets/DSEC/train_semantic_segmentation/train', event_dir='/media/geoffroy/T7/Thèse/Datasets/DSEC/train_events/')

print(f"Dataset has {len(dt)} elements")

idx = 0
events, label = dt[idx]
print(events.keys())
print(label)

from visualisation import plot_event_frame, plot_semantic_label
dim = (max(events['y'] + 1), max(events['x'] + 1))
print(dim)

plt.figure()
plt.subplot(1,2,1)
plot_event_frame(events, dim)
plt.title("Event Histogram")

plt.subplot(1,2,2)
plot_semantic_label(label)
plt.title("Semantic Segmentation Label")
plt.show()

