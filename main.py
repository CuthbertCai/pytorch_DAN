import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pylab
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import utils
import models
import params
import train, test

src_train_dataloader = utils.get_train_loader('MNIST')
src_test_dataloader =utils.get_test_loader('MNIST')
tgt_train_dataloader = utils.get_train_loader('MNIST_M')
tgt_test_dataloader = utils.get_test_loader('MNIST_M')

common_net = models.Extractor()
src_net = models.Classifier()
tgt_net = models.Classifier()

src_dataiter = iter(src_train_dataloader)
tgt_dataiter = iter(tgt_train_dataloader)
src_imgs, src_labels = next(src_dataiter)
tgt_imgs, tgt_labels = next(tgt_dataiter)

src_imgs_show = src_imgs[:4]
tgt_imgs_show = tgt_imgs[:4]

utils.imshow(vutils.make_grid(src_imgs_show))
utils.imshow(vutils.make_grid(tgt_imgs_show))

train_hist = {}
train_hist['Total_loss'] = []
train_hist['Class_loss'] = []
train_hist['MMD_loss'] = []

test_hist = {}
test_hist['Source Accuracy'] = []
test_hist['Target Accuracy'] = []

if params.use_gpu:
    common_net.cuda()
    src_net.cuda()
    tgt_net.cuda()

src_features = common_net(Variable(src_imgs.expand(src_imgs.shape[0], 3, 28, 28).cuda()))
tgt_features = common_net(Variable(tgt_imgs.expand(tgt_imgs.shape[0], 3, 28, 28).cuda()))
src_features = src_features.cpu().data.numpy()
tgt_features = tgt_features.cpu().data.numpy()
src_features = TSNE(n_components= 2).fit_transform(src_features)
tgt_features = TSNE(n_components= 2).fit_transform(tgt_features)

plt.scatter(src_features[:, 0], src_features[:, 1], color = 'r')
plt.scatter(tgt_features[:, 0], tgt_features[:, 1], color = 'b')
plt.title('Non-adapted')
pylab.show()

optimizer = optim.SGD([{'params': common_net.parameters()},
                       {'params': src_net.parameters()},
                       {'params': tgt_net.parameters()}], lr= params.lr, momentum= params.momentum)

criterion = nn.CrossEntropyLoss()

for epoch in range(params.epochs):
    t0 = time.time()
    print('Epoch: {}'.format(epoch))
    train.train(common_net, src_net, tgt_net, optimizer, criterion,
                epoch, src_train_dataloader, tgt_train_dataloader, train_hist)
    t1 = time.time() - t0
    print('Time: {:.4f}s'.format(t1))
    test.test(common_net, src_net, src_test_dataloader, tgt_test_dataloader, epoch, test_hist)

src_features = common_net(Variable(src_imgs.expand(src_imgs.shape[0], 3, 28, 28).cuda()))
tgt_features = common_net(Variable(tgt_imgs.expand(tgt_imgs.shape[0], 3, 28, 28).cuda()))
src_features = src_features.cpu().data.numpy()
tgt_features = tgt_features.cpu().data.numpy()
src_features = TSNE(n_components= 2).fit_transform(src_features)
tgt_features = TSNE(n_components= 2).fit_transform(tgt_features)


utils.visulize_loss(train_hist)
utils.visualize_accuracy(test_hist)
plt.scatter(src_features[:, 0], src_features[:, 1], color = 'r')
plt.scatter(tgt_features[:, 0], tgt_features[:, 1], color = 'b')
plt.title('Adapted')
pylab.show()


