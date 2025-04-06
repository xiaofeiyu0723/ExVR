import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm 

from mobileposer.config import *
from mobileposer.helpers import * 
import mobileposer.articulate as art
from mobileposer.constants import MODULES
from mobileposer.utils.model_utils import load_model
from mobileposer.data import PoseDataset
from mobileposer.models import MobilePoserNet


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps)

    def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        tran_p = tran_p.clone().view(-1, 3)
        tran_t = tran_t.clone().view(-1, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                                  'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)', 
                                  'Jitter Error (100m/s^3)', 'Distance Error (cm)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


@torch.no_grad()
def evaluate_pose(model, dataset, num_past_frame=20, num_future_frame=5, evaluate_tran=False):
    # specify device
    device = model_config.device

    # load data
    xs, ys = zip(*[(imu.to(device), (pose.to(device), tran)) for imu, pose, joint, tran in dataset])

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    offline_errs, online_errs = [], []
    tran_errors = {window_size: [] for window_size in list(range(1, 8))}
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(list(zip(xs, ys))):
            model.reset()
            pose_p_offline, joint_p_offline, tran_p_offline, _ = model.forward_offline(x.unsqueeze(0), [x.shape[0]])
            pose_t, tran_t = y
            pose_t = art.math.r6d_to_rotation_matrix(pose_t)

            if getenv("ONLINE"):
                online_results = [model.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
                pose_p_online, joint_p_online, tran_p_online, _ = [torch.stack(_)[num_future_frame:] for _ in zip(*online_results)]

            if evaluate_tran:
                # compute gt move distance at every frame 
                move_distance_t = torch.zeros(tran_t.shape[0])
                v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
                for j in range(len(v)):
                    move_distance_t[j + 1] = move_distance_t[j] + v[j] # distance travelled

                for window_size in tran_errors.keys():
                    # find all pairs of start/end frames where gt moves `window_size` meters
                    frame_pairs = []
                    start, end = 0, 1
                    while end < len(move_distance_t):
                        if move_distance_t[end] - move_distance_t[start] < window_size: # if not less than the window_size (in meters)
                            end += 1
                        else:
                            if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                frame_pairs.append((start, end))
                            start += 1

                    # calculate mean distance error 
                    errs = []
                    for start, end in frame_pairs:
                        vel_p = tran_p_offline[end] - tran_p_offline[start]
                        vel_t = (tran_t[end] - tran_t[start]).to(device)
                        errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                    if len(errs) > 0:
                        tran_errors[window_size].append(sum(errs) / len(errs))

            offline_errs.append(evaluator.eval(pose_p_offline, pose_t, tran_p=tran_p_offline, tran_t=tran_t))
            if getenv("ONLINE"):
                online_errs.append(evaluator.eval(pose_p_online, pose_t, tran_p=tran_p_online, tran_t=tran_t))

    # print joint errors
    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    if getenv("ONLINE"):
        print('============== online ================')
        evaluator.print(torch.stack(online_errs).mean(dim=0))
    
    # print translation errors
    if evaluate_tran:
        print([0] + [torch.tensor(_).mean() for _ in tran_errors.values()])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dip')
    args = parser.parse_args()

    # load model 
    model = load_model(args.model)

    # load dataset
    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Test dataset: {args.dataset} not found.")
    dataset = PoseDataset(fold='test', evaluate=args.dataset)

    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    evaluate_pose(model, dataset)
