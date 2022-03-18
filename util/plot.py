import torch
import visdom
import numpy as np

def plot_diff_pcds(pcds, vis, title, legend, win=None):
    '''
    :param pcds: python list, include pcds with different size
    :      legend: each pcds' legend
    :return:
    '''
    device = pcds[0].device
    assert vis.check_connection()

    pcds_data = torch.Tensor().to(device)
    for i in range(len(pcds)):
        pcds_data = torch.cat((pcds_data, pcds[i]), 0)

    pcds_label = torch.Tensor().to(device)
    for i in range(1, len(pcds) + 1):
        pcds_label = torch.cat((pcds_label, torch.Tensor([i] * pcds[i - 1].shape[0]).to(device)), 0)

    vis.scatter(X=pcds_data, Y=pcds_label,
                opts={
                    'title': title,
                    'markersize': 3,
                    # 'markercolor': np.random.randint(0, 255, (len(pcds), 3)),
                    'webgl': True,
                    'legend': legend},
                win=win)


def plot_loss_curves(runner, losses, vis, win=None):
    if not hasattr(runner, 'curve_data'):
        runner.curve_data = {'X':[], 'Y':[], 'legend':list(losses.keys())}
    runner.curve_data['X'].append(runner.epoch)
    runner.curve_data['Y'].append([losses[k] for k in runner.curve_data['legend']])
    
    vis.line(
        X=np.array(runner.curve_data['X']),
        Y=np.array(runner.curve_data['Y']),
        opts={
            'title': 'runing loss over time',
            'legend': runner.curve_data['legend'],
            'xlabel': 'epoch',
            'ylabel': 'loss'},
        win=win)