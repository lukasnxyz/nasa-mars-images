import os
from PIL import Image
import csv
import numpy as np
from tqdm import tqdm

# load labels
labels_path = 'data/hirise-map-proj-v3_2/labels-map-proj_v3_2.txt'
labels_dict = {}
with open(labels_path) as labels_file:
  for line in tqdm(labels_file, desc='Processing labels'):
    parts = line.split()
    if len(parts) == 2:
      filename, label = parts
      labels_dict[filename] = int(label)
print(f'num of labels: {len(labels_dict)}')

dataset_path = 'data/hirise-map-proj-v3_2/map-proj-v3_2'
data = []

# reshape+flatten images, associate each with a label, pixel val from 0.0-1.0
for filename in tqdm(os.listdir(dataset_path), desc='Converting images'):
  if filename.endswith('.jpg'):
    img_path = os.path.join(dataset_path, filename)
    img = Image.open(img_path).convert('L')
    img = img.resize((50, 50)) # maybe something smaller?
    img_flattened = np.array(list(img.getdata()), dtype=np.float32)
    img_flattened = img_flattened/img_flattened.max()
    img_flattened = np.insert(img_flattened, 0, labels_dict[filename])
    data.append(img_flattened)
print(f'images transformed: {len(data)}')

# save images/labels to csv file
output_csv = 'data/nasa_mars_images.csv' 
with open(output_csv, mode='w', newline='') as csv_file:
  writer = csv.writer(csv_file)
  print('saving transformed images')
  writer.writerows(data)