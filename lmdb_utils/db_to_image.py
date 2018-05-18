import os
import sys
import lmdb
import numpy as np
import cv2
import time

def convert_back(db_path):
	

	env = lmdb.open(db_path, max_readers = 10, readonly=True)
	with env.begin(write=False) as txn:
		for k,v in txn.cursor():
			print("Current Key : {}".format(k))

			img=cv2.imdecode(np
				.fromstring(v, dtype=np.uint8),1)
			save_dir= '/home/trainee/converted/'+str(k)+'.png'
			cv2.imwrite(save_dir, img)
			



if __name__== '__main__':
	arg1 = sys.argv[1]
	start_time = time.time()
	convert_back(arg1)
	stop_time = time.time()
	print("Total conversion time : {}".format(stop_time - start_time))