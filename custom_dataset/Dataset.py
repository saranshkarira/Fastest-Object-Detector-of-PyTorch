import torch.utils.data as data
from PIL import Image

class MODClass(data.Dataset):
	def __init__(self, db_path, transform=None, target_transform=None):
		import lmdb
		# self.db_path = os.path.expanduser(db_path)
		self.transform = transform # What transforms?
		self.target_transform = target_transform

		self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
		with self.env.begin(write=False) as txn:
			with txn.cursor() as cursor:
				self.length = txn.stat()['entries']
				self.data = iter(cursor) #takes all the db into ram at once at superspeed, if db bigger than ram than add cursor slicing
			
		def __getitem__(self, index):
		img,target = None, None
		
		img_buf = self.data[index]#accessing key
		buf = six.BytesIO()
		buf.write(img_buf)
		buf.seek(0)
		img = Image.open(buf).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)

		#define target
		if self.target_transform is not None:
			target = self.target_transform(target)


			return img, target

		def __len__(self):
			return self.length

		def __repr__(self):
			return self.__class__.__name__+' ('+self.db_path+')'


class MOD(data.Dataset):  #Miliatry Object Detection Dataset
	"""
	Args:
		db_path : path to train and test
		classes (string or list): List of categories to load
		transform(optional): takes in an PIL image and returns a transformed version
		target_transform : transforms the target
	"""
	def __init__(self, db_path, transform=None, target_transform=None, categories=custom):
		categories = self.categories #3 categories coco, pascal, custom
		dset_opts = ['train','val','test']
		self.db_path = os.path.expanduser(db_path)
		self.transform = transform
		self.target_transform = target_transform

		

		


