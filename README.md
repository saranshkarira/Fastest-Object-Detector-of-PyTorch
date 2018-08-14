# Fastest Object Detection on PyTorch
This YOLOv2 based API is a fast(fastest?), consistent and robust solution to train your own object detector with your own custom dataset from scratch including annotating part.

**NOTE** : While the basic underlying API is made robust and complete, there are couple more lines of development that can be pursued with sure success which will make it even more faster, lighter and better.
I'm currently indulged in research so further development will be slower but in case you're interested in developing this project, you're welcome to contact me to discuss these lines of development.

## Why this API?
The API in its current state is only fast in terms of training, not detection, so why use it at all? Or a better question, why make it at all?
Well, being a part of a third-world country with limited resources and unaffordable clouds due to currency conversion, time is a very precious commodity, 
all I could get my hands on were a couple of K80s with time restrictions, so I made this API keeping those things in mind. Since then, I've learned and implemented a lot more techniques in other projects that can make this API even faster 
unfortunately I really don't have time and resources to integrate them into this one.

## Features:

### Custom Dataset/Transfer Learning:
Use this API if you want to make your own dataset and train the pretrained YOLOv2 over it.

### Speed and Accuracy:
The basic functions of the code were inspired from https://github.com/longcw/yolo2-pytorch which is still an experimental project and wasn't achieving mAP as per the orignal YOLOv2 implementation, yet I found it impressive that the postprocessing functions and custom modules were implemented in Cython, using them certainly gave a headstart, plus I improved it even more so now this API is 700% faster than that implementation and is as accurate as the orignal implementation.

### LMDB Support: 
- The LMDB database offers the fastest read speeds, is memory-mapped and corruption proof.
- This is the most efficient way to integrate LMDB with PyTorch that you'll ever find on web (And an even more efficient technique is on it's way :D)

### Matlab ImageLabeller Support:
- I tried and tested each and every bounding box annotator available out there(For Linux), Matlab's ImageLabeller was hands-down the fastest, most robust tool. Sad that it was usable in Matlab only and had a complex structure efficient to Matlab only, well sad no more. Now you can convert all those annotations altogether into a JSON file in the blink of an eye.

## Tensorboard Support : 
- I can't stress this enough, it's extremely weary to stare at terminal or plotting different graphs by the code and maintaining proper consistency, with TensorBoard, never again.

## Ease of Use and Robustness : 
- Once setup, the whole training just needs one single command and one optional flag. The API takes care of consistency (whether it's a new experiment or an old one, therefore where to save the TFlogs, checkpoints)

- Python2.7
- Cython 0.27.2
- OpenCV 3.3
- h5py
- TensorboardX
- LMDB
- Matlab R2017b 
- PyTorch 0.3

## Current bug: 
The object detector produces different number of boxes for different images due to this the mini-batch is made from custom collate function which is a list of torch tensors of different dimensions, this works fluently on single GPU but when it's loaded on multi-gpu using DataParallel class, the passed list to Darknet class(inherited from nn.Module) is actually being passed as None inside the class

## ToDo:
[x] Implementing LMDB in an efficient way.
[x] Matlab Script to export annotations into JSON
[x] Custom Dataset in Pytorch(including custom collate fxn)
[x] Multithreading
[x] Applying transfomations to Image as well as Annotations
[x] Training Regime
[x] Speeding up Training Regime
[x] TensorBoard
[x] Adding log,load and save consistency over multiple pause and plays.
[x] Only maintaining 'n' no of checkpoints
[x] Eval Code
[ ] Detection Pipeline using OpenCV
[ ] Support for PyTorch 0.4
[ ] A port to Python3
[ ] Add a parser function so that it can parse different flavors of YOLO using cfg files
[ ] Add a subparser function so the base DarkNet can be replaced with other classification nets such as MobileNets, Squeezenets etc.
[ ] Replace SGD with AdamW so the training speed increases even more
[ ] Add support for standard datasets such as COCO and VOC
[ ] Add options for several pruning methods.

