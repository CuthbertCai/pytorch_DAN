## Pytoch_DAN
> This is a simple implementation of [Learning Transferable Features with Deep Adaptation  
> Networks][1] with pytorch. This paper introduced a simple and effective method for  
> accomplishing domian adaptation with MMD loss. According to this paper,  
> multi-layer features are adapted with MMD loss. In this paper, model is based  
> on AlexNet and tested on several datasets, while this work just utilizes  
> LeNet and tests on MNIST and MNIST_M datasets. The original implementation  
> in caffe is [here][2].
  

### Data
> In this work, MNIST and MNIST_M datasets are used in experiments. MNIST dataset  
> can be downloaded with `torchvision.datasets`. MINIST_M dataset can be downloa-  
> ded at [Yaroslav Ganin's homepage][3]. Then you can extract the file to your data dire-  
> ctory and run the `preprocess.py` to make the directory able to be used with  
> `torchvision.datasets.ImageFolder`:
```
python preprocess.py
```
> If you could not download MNIST_M dataset from [Yaroslav Ganin's homepage][3], you cou-  
> ld download it from [MEGA Cloud][4]. Once you download it, then you just need to unzip  
> the file to `/data` and the `preprocess.py` should not be used.

### Experiments
> You can run `main.py` to implements the MNSIT experiments. This work's results  
> are as follows:  

|Method     | Target Acc(this work)|
|:----------:|:----------------:|
|Source Only| 0.5189|
|DAN       | 0.5829|



[1]:https://arxiv.org/pdf/1502.02791.pdf
[2]:https://github.com/thuml/DAN
[3]:http://yaroslav.ganin.net/
[4]:https://mega.nz/#!FuIQhYKJ!IVxcmZK1ZH2MZA-7gKkgR4FExPuJl7-m89eDRPVKhF4
