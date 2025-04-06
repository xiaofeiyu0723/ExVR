r"""
    Function to test a network.
"""

__all__ = ['test', 'test_next_batch']

import torch
import os


@torch.no_grad()
def test(net, test_dataloader, load_dir=None):
    r"""
    Test a net on the full dataset. Return input, output, label. Assuming input, output, label are tensors.

    :param net: Network to test.
    :param test_dataloader: Test dataloader, enumerable and has __len__. It loads (test_data, test_label) pairs.
    :param load_dir: If not None, network best weights are loaded before test.
    :return: Data, prediction and label.
    """
    if load_dir is not None:
        net.load_state_dict(torch.load(os.path.join(load_dir, 'best_weights.pt')))
    net.eval()
    data, pred, label = [torch.cat(_) for _ in zip(*[(d, net(d), l) for d, l in test_dataloader])]
    return data, pred, label


@torch.no_grad()
def test_next_batch(net, test_dataloader, load_dir=None):
    r"""
    Test a net on the next batch of data. Return input, output, label.

    :param net: Network to test.
    :param test_dataloader: Test dataloader, enumerable and has __len__. It loads (test_data, test_label) pairs.
    :param load_dir: If not None, network weights are loaded before test.
    :return: Data, prediction and label.
    """
    if load_dir is not None:
        net.load_state_dict(torch.load(os.path.join(load_dir, 'best_weights.pt')))
    net.eval()
    data, label = next(iter(test_dataloader))
    pred = net(data)
    return data, pred, label
