# Fastest Object Detection on PyTorch
- This YOLOv2 based API is a robust, consistent and fastest solution to train your own object detector with your own custom dataset from scratch including annotating the data.
- Also it delivers the fastest train and detect time speeds for PyTorch as well.
- Use this API if you want to train your object detector on your own custom data and classes from ground up.

**NOTE** : 

- I'm currently indulged in other activity so further development will be slower but in case you're interested in contributing to this project, you're welcome to contact me to discuss future lines of development.

## Why this API?

- Well, at my place, I have very limited resources and unaffordable clouds due to currency conversion, time is a very precious commodity.

- All I could get my hands on was a K40 with time restrictions per day, so I made this API keeping those things in mind(i.e. consistency in Pause/Play to train it over multiple days, train it in as less time as possible etc). 

- Since then, I've learned and implemented a lot more techniques in other projects that can make this API even faster 
unfortunately I really don't have time and resources to integrate them into this one.

## Features :

### Custom Dataset/Transfer Learning :
- Use this API if you want to make your own dataset and train the pretrained YOLOv2 over it.

### Speed and Accuracy :
- The basic functions of the code were inspired from https://github.com/longcw/yolo2-pytorch which is still an experimental project and wasn't achieving mAP as per the orignal YOLOv2 implementation.
- What I found impressive was that the postprocessing functions and custom modules were implemented in Cython, using them certainly gave a headstart, plus I improved it even more so now this API is 700% faster than longcw's and is as accurate as the orignal implementation.

### LMDB Support : 
- The LMDB database offers the fastest read speeds, is memory-mapped and corruption proof.
- This is the most efficient way to integrate LMDB with PyTorch that you'll ever find on web (And an even more efficient technique is on it's way :D)

### Matlab ImageLabeller Support : 
- I tried and tested each and every bounding box annotator available out there(For Linux), Matlab's ImageLabeller was hands-down the fastest, most robust tool. 
- Sad that it was usable in Matlab only and had a complex structure efficient to Matlab only.
- Well sad no more. This API will extract and convert all those annotations into a JSON file in the blink of an eye.

### Tensorboard Support : 
- I can't stress this enough, it's extremely weary to stare at terminal or plotting different graphs by the code and maintaining proper consistency, with TensorBoard, never again.

### Ease of Use and Robustness : 
- Once setup, the whole training just needs one single command and one optional flag. The API takes care of consistency (whether it's a new experiment or an old one, therefore where to save the TFlogs, checkpoints)

## Requirements :
- Python2.7
- Cython 0.27.2
- OpenCV 3.3
- h5py
- TensorboardX
- LMDB
- Matlab R2017b 
- PyTorch 0.3

## How to Use :

### Installation :
- Clone this repository
  ```bash
  https://github.com/saranshkarira/pytorch-yolov2-fastest.git
  ```
- Open ```make.sh```
```nano make.sh```
- Replace sm_35 @ -arch=sm_35 to the sm of your own GPU ([`Refer`](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))
- Build the reorg layer ([`tf.extract_image_patches`](https://www.tensorflow.org/api_docs/python/tf/extract_image_patches))
  ```bash
  cd pytorch-yolov2-fastest
  ./make.sh
  ```
- Make models and db directory
  ```bash
  mkdir -p models db
  ```
- Download the trained model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU) and put it in the models/ directory.

### Annotation :
1. Annotate Images using the Matlab ImageLabeller Tool.
2. Export the session as .mat file.
3. You can scale this to multiple people on multiple machines for a larger dataset, then export all these sessions as .mat file.
4. Put all these files in mat_to_json folder and run the encoder.m script.
5. You will have a single, nice and lightweight JSON file containing all the annotations.

### Preparing the dataset:
This step will pack your image dataset into a nice LMDB database file.
1. Put all the images in ```./db/raw/```
2. There might be some images in the dataset that you did not annotate because object was vague or simply not there.
3. Don't worry, put the annotation file(JSON) in ```./db/targets``` and now it will only pack images that you annotated.
4. You can get away with keeping the targets folder empty, in which case it will pack all the images in raw folder but you'll have to put JSON in targets anyway so I'd say do this before.
5. Run 
    ```bash
    python2.7 cursor_putting.py
    ```
6. LMDB files will be generated in ```.db/image_data/``` directory.
7. Now, for the final step, open ```.cfgs/config_voc.py``` and replace *VOC* classes with your custom classes.

### Training:
- To adjust your hyper-parameters, refer to this file:
```.cfgs/exps/darknet19_exp1.py```

Now, once everything is setup, you can train the model by:
1. If it's the first time, you'll need to activate *Transfer Learning*, and the model will be loaded from the pretrained model.
```bash
   python2.7 trainer.py -tl True
```
2. Now everytime after that, when you stop the training and want to resume it from the latest checkpoint.
```bash
   python2.7 trainer.py
```

**Note** : No need to worry about number of checkpoints, after every few epochs, old ones will be pruned.

### Tensorboard:
Run
  ```bash
  tensorboard --logdir .models/logs
  ```
### Results:
This is a trainer API, after the training is complete, you can use the .h5 file in various available YOLOv2 detection APIs and it will work all the same, especially longcw's or you can contribute or wait until a detection pipeline is made as well.


## Navigating the API
*For contributors only*
I believe, a good navigation goes a long way and speeds up the development process of a project.
The API is structured in the following way:

1. All the configuration files are in ```cfg/``` directory, general functions and settings(directory locations) are saved in ```config.py```, Dataset specific settings(Anchors, class_names) are saved in ```config_voc.py```, Experiment specific settings(hyper-parameters) are saved in ```exps/```

2. ```layer/``` contains custom modules written in cython that are compiled using make.sh script.
3. ```mat_to_json``` contains script to convert and export .mat annotation file into json.
4. ```db/``` contains 3 folders, ```raw```, ```image_data``` and ```targets```, for raw images, lmdb data and annotations respectively, basically all the dataset related stuff.
5. ```models``` contains model related stuff, training checkpoints, pretrained model and Tensorboard logs.
6. ```utils``` contains misc functions such as NMS, converting annotations to grid respective, saving and loading weights.
7. cursor_putting.py converts raw_images into lmdb database.
8. dataset.py is a custom pytorch dataset class with custom collate function. Also some eval code.
9. darknet.py has the model definition.
10. loss.py calculates the loss.
11. trainer.py is the main/central file that uses all the above files and trains the model.

## TO-DO:
- [x] Implementing LMDB in an efficient way.
- [x] Matlab Script to export annotations into JSON
- [x] Custom Dataset in Pytorch(including custom collate fxn)
- [x] Multithreading
- [x] Applying transfomations to Image as well as Annotations
- [x] Training Regime
- [x] Speeding up Training Regime
- [x] TensorBoard
- [x] Adding log,load and save consistency over multiple pause and plays.
- [x] Only maintaining 'n' no of checkpoints
- [x] Transfer Learning
- [x] Eval Code
- [x] DataParallel Issue
- [ ] Detection Pipeline using OpenCV
- [ ] Support for PyTorch 0.4
- [ ] A port to Python3
- [ ] Add a parser function so that it can parse different flavors of YOLO using cfg files
- [ ] Add a subparser function so the base DarkNet can be replaced with other classification nets such as MobileNets, Squeezenets etc.
- [ ] Replace SGD with AdamW so the training speed increases even more
- [ ] Add support for standard datasets such as COCO and VOC
- [ ] Add options for several pruning methods.

