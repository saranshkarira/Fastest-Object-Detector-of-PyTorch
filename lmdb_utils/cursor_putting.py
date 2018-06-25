
import os
import sys
import lmdb
import cv2
import glob


def write_to_db(env, batch):
    with env.begin(write=True) as txn:
        with txn.cursor() as cursor:
            cursor.putmulti(batch, dupdata=True, overwrite=True, append=False)


def converter(input_path, output_path, targets):
    env = lmdb.open(output_path, map_size=9959123412)
    batch = []
    counter = 0
    if os.path.exists(glob.glob(os.path.join(targets, '*.json'))[0]):
        target_file = glob.glob(os.path.join(targets, '*.json'))[0]
        with open(target_file) as opener:
            targets = json.load(opener)
        joker = [targets[i]['Var1'].split('/')[-1] for i in range(len(targets))]

    else:
        joker = os.listdir(input_path)

    joker.sort()

    for image_name in joker:
        print(counter)
        image_path = os.path.join(input_path, image_name)
        if not os.path.isfile(image_path):
            print("{} is not a file".format(image_name))

            continue
        img = cv2.imread(image_path)

        image_binary = cv2.imencode(".jpg", img)[1].tostring()

        imagekey = str.encode(image_name)

        batch.append((imagekey, image_binary))

        if counter % 500 == 0:
            write_to_db(env, batch)
            print("flushing cache")
            batch = []
        counter += 1

    write_to_db(env, batch)


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db"))
    input_path = os.path.join(path, "raw_images")
    output_path = os.path.join(path, "image_data")
    target_path = os.path.join(path, "targets")

    converter(input_path, output_path, target_path)
