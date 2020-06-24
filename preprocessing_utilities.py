import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py


DATADIR = r'.\Datasets\ISLR_Dataset'
DATA_BINDIR = r'.\Datasets\Binary_Dataset'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
			  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V',
			  'W', 'X', 'Y', 'Z']


#utility to convert dataset of images into a single channel(Grayscale) format
def cvt_dataset_into_bin():
	for category in CATEGORIES:
		bin_path = os.path.join(DATA_BINDIR, category)
		os.makedirs(bin_path)
		path = os.path.join(DATADIR, category)
		for image in os.listdir(path):
			img = cv2.imread(os.path.join(path, image))
			if(img is None):
				os.remove(os.path.join(path, image))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (128, 128))
			# plt.imshow(img, cmap='binary')
			# plt.show()
			cv2.imwrite(os.path.join(bin_path, image), img)
			print(os.path.join(bin_path, image), img)

			
#utility to load images and convert into a h5 file
def load_dataset():
	data = []
	labels = []
	for category in CATEGORIES:
		for img in os.listdir(os.path.join(DATA_BINDIR, category)):
			image = cv2.imread(os.path.join(DATA_BINDIR, category, img), 1)
			label = category
			data.append(image)
			labels.append(label)

	label = [n.encode('ascii', 'ignore') for n in labels]
	with h5py.File('Images_Labels_color.hdf5', 'w') as f:
		f.create_dataset('images', data=data)
		f.create_dataset('labels', data=label)

	f.close()
	return data, labels


