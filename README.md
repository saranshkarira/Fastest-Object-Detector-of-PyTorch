# Fastest Object Detection on PyTorch
This YOLOv2 based API is a fast(fastest?), consistent and robust solution to train your own object detector with your own custom dataset from scratch including annotating part.

**NOTE** : While the basic underlying API is made robust and complete, there are couple more lines of development that can be pursued with sure success which will make it even more faster, lighter and better.
I'm currently indulged in research so further development will be slower but in case you're interested in developing this project, you're welcome to contact me to discuss these lines of development.

## Why this API?
The API in its current state is only fast in terms of training, not detection, so why use it at all? Or a better question, why make it at all?
Well, being a part of a third-world country with limited resources and unaffordable clouds due to currency conversion, time is a very precious commodity, 
all I could get my hands on were a couple of K80s with time restrictions, so I made this API keeping those things in mind. Since then, I've learned and implemented a lot more techniques in other projects that can make this API even faster 
unfortunately I really don't have time and resources to integrate them into this one.

You can annotate your images at blazing speed, the images will be loaded into the LMDB which has been integrated with Pytorch in a novel way, a new algorithm is on it's way which will slash this reading and augmentation time even more removing any possiblity of I/O bounds.

## Current Use Case:
Use this API to train your custom detector from scratch and then use the learned model and load it into some other API for detections

## Future Prospects: 
To add support for other object detectors and load any family of classifier as underlying feature extractor so novel combinations of architectures could be made on the go.

## Current bug: 
The object detector produces different number of boxes for different images due to this the batch is made from custom collate function which is a list of torch tensors of different dimensions, this works fluently on single GPU but when it's loaded on multi-gpu using DataParallel class, the passed list to Darknet class(inherited from nn.Module) is actually being passed as None inside the class
