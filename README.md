## Pytoch_DAN
> This is a implementation of [Learning Transferable Features with Deep Adaptation  
> Networks][1] with pytorch. This paper introduced a simple and effective method for  
> accomplishing domian adaptation with MMD loss. According to this paper,  
> multi-layer features are adapted with MMD loss. In this paper, model is based  
> on AlexNet, while this work just utilizes LeNet and tests on MNIST and  
> MNIST_M datasets. The original implementation in caffe is [here][2].  

### Data
> In this work, MNIST and MNIST_M datasets are used in experiments. MNIST dataset  
> can be downloaded with `torchvision.datasets`. MINIST_M dataset can be downloa-  
> ded at [Yaroslav Ganin's homepage][3]. Then you can extract the file to your data dire-  
> ctory and run the `preprocess.py` to make the directory able to be used with  
> `torchvision.datasets.ImageFolder`:
```
python preprocess.py
```

### Experiments
> You can run `main.py` to implements the MNSIT experiments. This work's results  
> are as follows:  

|Method     | Target Acc(this work)|
|:----------:|:----------------:|
|Source Only| 0.5189|
|DANN       | 0.5829|



[1]:https://arxiv.org/pdf/1502.02791.pdf
[2]:https://github.com/thuml/DAN
[3]:http://yaroslav.ganin.net/