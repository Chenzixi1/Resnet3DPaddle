import os
import numpy as np

video_src_src_path = 'data/UCF-101-jpg'
label_name = os.listdir(video_src_src_path)
label_dir = {}
index = 0

label_name.sort()
for i in label_name:
    if i.startswith('.'):
        continue
    label_dir[i] = index
    index += 1
np.save('label_dir.npy', label_dir)
print(label_dir)
