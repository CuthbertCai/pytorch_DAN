import torch
from torch.autograd import Variable
import numpy as np

import params
import utils

def test(common_net, src_net, source_dataloader, target_dataloader, epoch, test_hist):

    common_net.eval()
    src_net.eval()

    source_correct = 0
    target_correct = 0

    for batch_idx, sdata in enumerate(source_dataloader):
        input1, label1 = sdata
        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)

        input1 = input1.expand(input1.shape[0], 3, 28, 28)
        output1 = src_net(common_net(input1))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

    for batch_idx, tdata in enumerate(target_dataloader):
        input2, label2 = tdata
        if params.use_gpu:
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)

        output2 = src_net(common_net(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

    source_accuracy = 100. * source_correct / len(source_dataloader.dataset)
    target_accuracy = 100. * target_correct / len(target_dataloader.dataset)

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'.format(
        source_correct, len(source_dataloader.dataset), source_accuracy,
        target_correct, len(target_dataloader.dataset), target_accuracy,
    ))
    test_hist['Source Accuracy'].append(source_accuracy)
    test_hist['Target Accuracy'].append(target_accuracy)
