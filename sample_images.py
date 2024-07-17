import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

imgs_file = 'data/nasa_mars_images.csv'
with open(imgs_file, mode='r') as file:
  reader = csv.reader(file)
  data = np.array([[float(item) for item in row] for row in tqdm(reader, desc='loading data')])

labels = data[:, 0]
imgs = data[:, 1:]

isi = np.random.randint(len(imgs), size=10)
for i in isi:
  print(f'index: {i}')
  plt.imshow(imgs[i].reshape((50, 50)))
  plt.title(labels[i])
  plt.show()