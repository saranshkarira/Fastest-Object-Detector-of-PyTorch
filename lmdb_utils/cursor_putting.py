


import os 
import sys
import lmdb
import cv2

def write_to_db(env, batch):
	with env.begin(write=True) as txn:
		with txn.cursor() as cursor:
			cursor.putmulti(batch, dupdata=True, overwrite=True, append=False)
		# for k,v in batch.items():
			
		# 	txn.put(k,v)



def converter(output_path):
	env = lmdb.open(output_path, map_size=9959123412)
	batch = {}
	counter=1
	for i in range(967):
		print(counter)
		image_name= "image"+str(counter)+".jpg"
		image_path= "/home/trainee/Desktop/Battlezone/People_JPG/" + image_name
		if not os.path.exists(image_path):
					print("{} does not exists".format(image_name))
					counter +=1
					continue
		img = cv2.imread(image_path)
		
		#print(img)
		image_binary = cv2.imencode(".jpg", img)[1].tostring()
		#print(type(string))
		#image_binary = str.encode(string)
		#print(image_binary)
		#break
		#with open(image_path, 'r') as f:
		#	image_binary = f.read()
		imagekey = str.encode(image_name)
		# arbitrarykey = str(counter)
		# batch[imagekey] = image_binary
		# batch[arbitrarykey] = i+(i^2)

		#Tuple
		batch = []
		batch.append((imagekey, image_binary))

		if  i == 966: #counter % 900 == 0
			write_to_db(env, batch)
			print("flushing cache")
		counter +=1
		

if __name__ == '__main__':
	output_path = sys.argv[1]
	converter(output_path)

