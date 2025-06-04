import numpy as np
import matplotlib.pyplot as plt

def plot_event_frame(event_stream,dim):
    w, h = dim
    frame = np.zeros((3,w,h))

    for i in range(len(event_stream['x'])):
        x = event_stream['x'][i]
        y = event_stream['y'][i]
        p = event_stream['p'][i]
        
        if p == 0: #OFF -> blue
            p = 2
        else:
            p = 0 #ON -> red

        frame[p,y,x] += 1
    
    for y in range(w):
        for x in range(h):
            if frame[0,y,x] >= frame[2,y,x]:
                frame[2,y,x] = 0
            else:
                frame[0,y,x] = 0

    frame = np.transpose(frame, (1,2,0))
    plt.imshow(frame)
    
def plot_semantic_label(label_path):
    frame = plt.imread(label_path)
    plt.imshow(frame)
