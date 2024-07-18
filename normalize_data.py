import csv
import numpy as np
from tqdm import tqdm

imgs_file = 'data/nasa_mars_images.csv'
with open(imgs_file, mode='r') as file:
  reader = csv.reader(file)
  data = np.array([[np.float32(item) for item in row] for row in tqdm(reader, desc='loading csv data')])

n_data = []
data_0 = []
for row in data:
  if row[0] != 0.0:
    n_data.append(row)
  else:
    data_0.append(row)

rnd_idxs = np.random.randint(len(data_0), size=2000)

for i in rnd_idxs:
  n_data.append(data_0[i])

output_csv = 'data/normalized_nasa_mars_images.csv'
with open(output_csv, mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerows(n_data)