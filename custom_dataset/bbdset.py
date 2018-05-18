##DataSet Barebones
## targets are treated as a [-1,2] matrix of points of polygon with 2 coordinates
#Imports
import torch.utils.data as data
import pandas as pd #what about hdf5
import os


#Dataclass
class dataset(data.Dataset):
	def __init__(self, csv_file, root_dir, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.target_file = pd.read_csv(csv_file) #Add hdf5 as default in future
        self.root_dir = root_dir
        self.transform = transform

##len
def __len__(self):
	return len(self.targets)
##getitem
def __getitem__(self, idx):
	img_name = os.path.join(self.root_dir, self.target_file.iloc[idx,0])
	image = io.imread(img_name)
	targets = self.target_file.iloc[idx,1:].as_matrix()
	targets = targets.astype('float').reshape()## add the reshape dims
	sample = {'image': image, 'targets': targets}

	if self.transform:
		sample = self.transform(sample)

	return sample

## transforms

#Rescale
class Rescale(object):
	"""Rescale the image in a sample to a given size.

	Args:
	    output_size (tuple or int): Desired output size. If tuple, output is
	        matched to output_size. If int, smaller of image edges is matched
	        to output_size keeping aspect ratio the same.
	"""
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
	def __call__(self, sample):
		image, targets = sample['image'], sample['targets']
		h,w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h>w:
				new_h, new_w = self.output_size*h/w, self.output_size

			else:
				new_h, new_w = self.output_size, self.output_size*w/h

		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        targets = targets*[new_w/w, new_h/h]

        return {'image': img, 'targets': targets}

#RandomCrop
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__{self, output_size):
    	assert isinstance(output_size, (int, tuple))
    	if isinstance(output_size, int):
    		self.output_size = (output_size, output_size)
    	else:
    		assert len(output_size) == 2
    		self.output_size = output_size

    def __call__(self, sample):
    	image, targets = sample['image'], sample['targets']

    	h, w = image.shape[:2]
    	new_h, new_w = self.output_size

    	top = np.random.randint(0,h -new_h)
    	left = np.random.randint(0,w - new_w)

    	image = image[top: top+new_h, left: left+new_w]

    	targets = targets - [left, top]

    	return {'image':image, 'landmarks':landmarks}

#ToTensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, targets = sample['image'], sample['targets']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'targets': torch.from_numpy(targets)}

