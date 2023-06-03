import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from utils import AverageMeter, ProgressMeter, accuracy


class CosineLinear(nn.Module):
    """
    A PyTorch module for a linear layer with cosine similarity.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        sigma (bool): Whether to include a learnable scaling factor.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        weight (nn.Parameter): The learnable weight tensor.
        bias (None): This module does not use a bias.
        sigma (Optional[nn.Parameter]): The learnable scaling factor tensor.
    """

    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class Temperature(torch.nn.Module):
    """
    A PyTorch module for temperature scaling during calibration.

    Attributes:
        temperature (nn.Parameter): The learnable temperature scaling factor.
    """

    def __init__(self):
        super(Temperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """
        Applies temperature scaling to the logits.

        Args:
            logits (torch.Tensor): The input logits.

        Returns:
            torch.Tensor: The logits after temperature scaling.
        """
        return logits / self.temperature


class BiC(nn.Module):
    """
    A PyTorch module for bias adjustment of new classes in a classification model.

    Args:
        num_new_cls (int): The number of new classes to adjust the bias for.

    Attributes:
        alpha (nn.Parameter): The learnable scaling factor for the new classes.
        beta (nn.Parameter): The learnable bias term for the new classes.
        num_new_cls (int): The number of new classes to adjust the bias for.
    """

    def __init__(self, num_new_cls):
        super(BiC, self).__init__()
        self.alpha, self.beta = nn.Parameter(
            torch.ones(1) * 0.75), nn.Parameter(torch.ones(1) * 0.001)
        self.num_new_cls = num_new_cls

    def forward(self, logits):
        """
        Adjusts the bias of the new classes in the logits.

        Args:
            logits (torch.Tensor): The input logits.

        Returns:
            torch.Tensor: The logits with adjusted bias for the new classes.
        """
        extra = logits.size(1) - self.num_new_cls
        old, new = logits[:, :extra], logits[:, extra:]
        new = self.alpha * new + self.beta
        outputs = torch.cat((old, new), dim=1)
        return outputs


def correct_weights(model, valloader, calibration_method, logger, expand_size):
    """
    Corrects the weights of the fully connected layer in a PyTorch model according to the WA calibration method.

    Args:
        model (nn.Module): The PyTorch model to correct the weights for.
        valloader (DataLoader): The validation data loader.
        calibration_method (str): The calibration method to use.
        logger (Logger): The logger object for logging.
        expand_size (int): The number of new classes to expand the model for.

    Raises:
        AssertionError: If `expand_size` is 0.

    Returns:
        None
    """
    # Weights corrected according to: Maintaining Discrimination and Fairness in Class Incremental Learning (https://arxiv.org/pdf/1911.07053.pdf)
    if calibration_method == 'WA':
        assert (expand_size != 0)
        extra = model.fc.weight.size(0) - expand_size
        gamma = model.fc.weight[:extra].norm(
            p=2, dim=1).mean() / model.fc.weight[extra:].norm(p=2,
                                                              dim=1).mean()
        with torch.no_grad():
            intermediate = model.fc.weight[extra:] * gamma
            model.fc.weight[extra:] = nn.Parameter(intermediate)
            if model.fc.bias is not None:
                model.fc.bias[extra:] = nn.Parameter(model.fc.bias[extra:] *
                                                     gamma)

    # Weights corrected according to: Large Scale Incremental Learning (https://arxiv.org/abs/1905.13260)
    elif calibration_method == 'BiC':
        assert (expand_size != 0)
        calibrator = BiC(num_new_cls=expand_size).cuda()
        calib_optimizer = Adam(calibrator.parameters(), lr=0.01)

        ### TODO: Optimize to consume tiny computational cost by optimizing on a small set of valset, reusing predictions
        for lr_exp in range(2, 4):
            # Assuming that 8 epochs each with a lr decay by 10 suffices to make the calibrator converge (loss does stop decreasing)
            for param_group in calib_optimizer.param_groups:
                param_group['lr'] = 0.1**(lr_exp)

            for epoch in range(5):
                calibrator, calib_optimizer = calibrate(
                    loader=valloader,
                    model=model,
                    calibrator=calibrator,
                    optimizer=calib_optimizer,
                    logger=logger,
                    epoch=epoch)

        extra = model.fc.weight.size(0) - expand_size
        with torch.no_grad():
            intermediate = model.fc.weight[extra:] * calibrator.alpha
            model.fc.weight[extra:] = nn.Parameter(intermediate)
            if model.fc.bias is not None:
                model.fc.bias[extra:] = nn.Parameter(model.fc.bias[extra:] *
                                                     calibrator.alpha +
                                                     calibrator.beta)

    # Baseline weight correction according to: On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    elif calibration_method == 'Temperature':
        assert (expand_size != 0)
        calibrator = Temperature().cuda()
        calib_optimizer = Adam(calibrator.parameters(), lr=0.01)

        ### TODO: Optimize for minimal computational overhead by ternary search, no sgd needed
        for lr_exp in range(2, 4):
            # Assuming that 8 epochs each with a lr decay by 10 suffices to make the calibrator converge (loss does stop decreasing)
            for param_group in calib_optimizer.param_groups:
                param_group['lr'] = 0.1**(lr_exp)

            for epoch in range(5):
                calibrator, calib_optimizer = calibrate(
                    loader=valloader,
                    model=model,
                    calibrator=calibrator,
                    optimizer=calib_optimizer,
                    logger=logger,
                    epoch=epoch)

        model.fc.weight = nn.Parameter(model.fc.weight /
                                       calibrator.temperature)
        if model.fc.bias is not None:
            model.fc.bias = nn.Parameter(model.fc.bias /
                                         calibrator.temperature)

    return model


def calibrate(loader, model, calibrator, optimizer, logger, epoch):
    """
    Calibrates a PyTorch model using a given calibrator module and optimizer.

    Args:
        loader (DataLoader): The data loader for calibration.
        model (nn.Module): The PyTorch model to calibrate.
        calibrator (nn.Module): The PyTorch module to use for calibration.
        optimizer (Optimizer): The optimizer to use for calibration.
        logger (Logger): The logger object for logging.
        epoch (int): The current epoch number.

    Returns:
        Tuple[nn.Module, Optimizer]: The calibrated model and optimizer.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(logger,
                             len(loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{0}] LR: [{1:.4f}]".format(
                                 epoch, optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.eval()
    calibrator.train()
    end = time.time()

    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(images)

        output = calibrator(logits)
        loss = nn.CrossEntropyLoss()(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display_summary()
    return calibrator, optimizer
