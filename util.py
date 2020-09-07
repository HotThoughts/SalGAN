import os
import imageio
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from model import *
from train import *
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams.update({'font.size': 15})
# plt.rc('figure', titlesize=22)

def read_file(filename):
	lines = []
	with open(filename, 'r') as file:
	    for line in file: 
	        line = line.strip() #or some other preprocessing
	        lines.append(line)
	return lines


class FixationDataset(Dataset):
	def __init__(self, root_dir, image_file, fixation_file, transform=None):
		self.root_dir = root_dir
		self.image_files = read_file(image_file)
		self.fixation_files = read_file(fixation_file)
		self.transform = transform
		assert(len(self.image_files) == len(self.fixation_files))

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
	 
		img_name = os.path.join(self.root_dir, self.image_files[idx])
		image = imageio.imread(img_name)

		fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
		fix = imageio.imread(fix_name)

		sample = {'image': image, 'fixation': fix}

		if self.transform:
			sample = self.transform(sample)

		return sample

def get_data(root_dir, bs):
    transform = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
    train_ds = FixationDataset(root_dir, f'{root_dir}train_images.txt', f'{root_dir}train_fixations.txt', transform)
    valid_ds = FixationDataset(root_dir, f'{root_dir}val_images.txt', f'{root_dir}val_fixations.txt', transform)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    return train_dl, valid_dl

class Rescale():
    '''Recaling data'''
    def __init__(self):
        pass
    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        image = image.astype(np.float32) / 255.0
        fixation = fixation.astype(np.float32) / 255.0
        return {'image': image, 'fixation': fixation}

class ToTensor():
    '''Convert data to a tensor'''
    def __init__(self):
        pass
    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        image = torch.from_numpy(np.transpose(image))
        fixation = torch.from_numpy(np.expand_dims(fixation, axis=0))
        return {'image': image, 'fixation': fixation}

class Normalize():
    '''Normalize tensor'''
    def __init__(self):
        pass
    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return {'image': normalize(image), 'fixation': fixation}


def save_plot(ls, name):
	'''Plot the image and predicted fixation map side by side'''
	N = len(ls)
	f = plt.figure(figsize=(10, 10))
	idx = 1
	for i, item in enumerate(ls):
		image, truth, prediction = item
		# Image
		f.add_subplot(N, 3, idx)
		plt.axis('off')
		plt.title("Image")
		plt.imshow(image)
		idx +=1
		# Groud truth
		f.add_subplot(N, 3, idx)
		plt.axis('off')
		plt.title('Ground Truth')
		plt.imshow(truth, cmap='gray')
		idx +=1
		# Predicted fixation map
		f.add_subplot(N, 3, idx)
		plt.axis('off')
		plt.title('Prediction')
		plt.imshow(prediction, cmap='gray')
		idx +=1
	plt.savefig(name)
	print(f'{name} saved.')

def plot_loss():
	'''Plot the progress of training and validation loss over epochs'''
	train_loss = []
	valid_loss = []
	for i in range(20):
		CK_PATH = f'checkpoint/exp02/epoch{i}.ckpt'
		checkpoint = torch.load(CK_PATH, map_location=torch.device('cpu'))
		train_loss.append(checkpoint['train_loss'])
		valid_loss.append(checkpoint['valid_loss'])
	x = range(len(train_loss))
	plt.figure(figsize=(10,6))
	plt.plot(x, train_loss, '-o', label = 'Training loss')
	plt.plot(x, valid_loss, '-o', label = 'Validation loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	# plt.xticks(x)
	plt.legend(loc = 'upper right')
	# plt.grid(True)
	plt.show()

def plot_adversarial_loss():
	'''Plot the progress of validation loss when training GAN'''
	loss_BCE = np.ones(10)
	loss_BCE_GAN = np.ones(120)
	for i in range(1, 131):
		CK_PATH = f'../SalGAN_checkpoint/exp02/epoch{i}.ckpt'
		checkpoint = torch.load(CK_PATH, map_location=torch.device('cpu'))
		if i < 11:
			loss_BCE[i-1] = checkpoint['valid_loss'].data.item()
		else:
			loss_BCE_GAN[i-11] = checkpoint['G_valid_loss'].data.item()
	x_BCE = range(len(loss_BCE))
	x_BCE_GAN = range(11, len(loss_BCE_GAN)+11)
	plt.figure(figsize=(10,6))
	plt.plot(x_BCE, loss_BCE, '-', color = 'b', label = 'BCE only')
	plt.plot(x_BCE_GAN, loss_BCE_GAN, '-', color = 'g', label = 'BCE + GAN')
	plt.xlabel('Epoch')
	plt.ylabel('Validation Loss')
	# plt.xticks(x)
	plt.legend(loc='upper right', frameon=False)
	# plt.grid(True)
	plt.savefig('SalGAN Loss.png')
	print('SalGAN Loss.png saved.')

def produce_and_save_test_fix(ROOT_DIR, CK_PATH):
	'''Produce fixation map for test images and save it to folder'''
	def load_and_transform(img_path):
		img = imageio.imread(img_path)
		img = img.astype(np.float32) / 255.0
		img = torch.from_numpy(np.transpose(img))
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		img = normalize(img)
		return img.unsqueeze(0)
	# Load image paths
	image_paths = read_file(f'{ROOT_DIR}test_images.txt')
	for img_path in image_paths:
		index = img_path[18:22]
		img_path = os.path.join(ROOT_DIR, img_path)
		img = load_and_transform(img_path)
		# Load model
		model = Generator()
		checkpoint = torch.load(CK_PATH, map_location=torch.device('cpu'))
		model.load_state_dict(checkpoint['G_state_dict']) 
		# Pass image to model
		out = model.predict(img)
		fix = out[0].squeeze(0).data.cpu().numpy() * 255.0
		# Save prediction
		matplotlib.image.imsave(f'../Wei_Yiyao_predictions_V2/prediction-{index}.png', fix, cmap='gray')
		print(f'prediction-{index}.png saved.')


def visualize_prediction(root_dir, CK_PATH, name):
	'''Plot original image, ground truth, and prediction in row'''
	model = Generator()
	checkpoint = torch.load(CK_PATH, map_location=torch.device('cpu'))
	# model.load_state_dict(checkpoint['model_state_dict']) 
	model.load_state_dict(checkpoint['G_state_dict'])


	transform = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
	valid_ds = FixationDataset(root_dir, f'{root_dir}val_images.txt', f'{root_dir}val_fixations.txt', transform)
	valid_dl = DataLoader(valid_ds, batch_size=1)

	img_idx = 3006

	ls = []
	for batch in valid_dl:
		img, fixation = batch['image'], batch['fixation']
		out = model.predict(img)

		prediction = out[0].squeeze(0).data.cpu().numpy() * 255.0
		truth = fixation[0].squeeze(0).data.cpu().numpy() * 255.0

		image = imageio.imread(f'{root_dir}/images/validation/image-{img_idx}.png')
		ls.append((image, truth, prediction))
		img_idx += 1

		if (img_idx > 3010): break

	save_plot(ls, f'{name}.png')

def main():
	produce_and_save_test_fix('/export/scratch/CV2/', '../SalGAN_checkpoint/exp02/epoch30.ckpt')
	# visualize_prediction('/export/scratch/CV2/', '../SalGAN_checkpoint/exp02/epoch30.ckpt', 'SalGAN_exp02_epoch30')
	# plot_adversarial_loss()

if __name__ == '__main__':
    main()
