import os
import torch
import json
import utils
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
from scipy.special import softmax
from collections import defaultdict
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from eval_detection import ANETdetection
from eval_frame import FrameDetection

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled = False


def post_process(mask):
    for i in range(0, len(mask) - 2):
        if mask[i] == mask[i + 2] and mask[i] != mask[i + 1]:
            mask[i + 1] = mask[i]


def generate_segment(prediction, action_score, threshold_type='mean', adjust_mean=1.0, frame_type='max', act_weight=1.0):
    '''
    generate proposal based score sequence
    Args:
        prediction: (N, num_class)
    Return:
        segments, frames
    '''
    segments = []
    frames = []
    topp = np.array([
        sorted(prediction[:, i], reverse=True) for i in range(len(prediction[0]))
    ])
    topp = np.transpose(topp, (1, 0))
    k = max(int(len(prediction) / 12), 1)
    cs = np.mean(topp[:k, :], axis=0)
    ind = (cs > np.max(cs) / 1.5) * (cs > 0)
    threshold = np.mean(prediction)
    for i in np.where(ind)[0]:
        score = prediction[:, i] + act_weight * action_score.squeeze()
        if threshold_type == 'mean':
            threshold = np.mean(score) * adjust_mean
        else:
            threshold = 0
        mask = (score > threshold).astype('float32')
        vid_pred = np.concatenate([np.zeros(1), mask, np.zeros(1)], axis=0)
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
        for j in range(len(s)):
            if frame_type == 'max':
                bias = np.argmax(score[s[j]:e[j]])
            else:
                bias = int((e[j] - s[j]) // 2)
            aggr_score = np.max(score[s[j]:e[j]]) + cs[i]
            frames.append([i, s[j] + bias, aggr_score])
            if e[j] - s[j] >= 2:
                segments.append([i, s[j], e[j], aggr_score])
    return segments, frames


def evaluate(itr,
             dataset,
             model,
             logger,
             groundtruth_filename,
             prediction_filename,
             device=torch.device('cuda'),
             background=True,
             subset='test',
             fps=25,
             stride=16,
             threshold_type='mean',
             frame_type='max',
             adjust_mean=1.0,
             act_weight=1.0,
             tiou_thresholds=np.linspace(0.1, 0.7, 7),
             use_anchor=False):
    '''
    generate proposals and evaluate
    '''
    with open(groundtruth_filename, 'r') as fr:
        gt_info = json.load(fr)['database']
    save_dict = {'version': dataset.dataset_name, 'external_data': 'None'}
    frame_dict = {'version': dataset.dataset_name, 'external_data': 'abc'}

    rs = defaultdict(list)
    rs2 = defaultdict(list)
    instance_logits_stack = []
    labels_stack = []

    inds = dataset.get_testidx()
    classlist = dataset.get_classlist()
    tps = [0] * len(classlist)
    aps = [0] * len(classlist)
    res = [0] * len(classlist)
    one_hots = np.eye(len(classlist))
    for idx in inds:
        feat = dataset.get_feature(idx)
        vname = dataset.get_vname(idx)
        duration = dataset.get_duration(idx)
        feat = torch.from_numpy(np.expand_dims(feat,
                                               axis=0)).float().to(device)
        frame_label = dataset.get_gt_frame_label(idx)
        video_label = dataset.get_video_label(idx)
        if len(video_label) < 1:
            continue
        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, att_logits_f, att_logits_r, att_logits = model(
                Variable(feat), device, is_training=False)
        logits_f, logits_r, tcam = logits_f[0], logits_r[0], tcam[0]
        topk = int(np.ceil(len(feat[0]) / 8))

        tmp = F.softmax(torch.mean(torch.topk(logits_f, k=topk, dim=0)[0],
                                   dim=0),
                        dim=0).cpu().data.numpy()
        tmp += F.softmax(torch.mean(torch.topk(logits_r, k=topk, dim=0)[0],
                                    dim=0),
                         dim=0).cpu().data.numpy()
        tmp += F.softmax(torch.mean(torch.topk(tcam, k=topk, dim=0)[0], dim=0),
                         dim=0).cpu().data.numpy()
        if background:
            tcam = tcam[:, 1:]
            tmp = tmp[1:]
        instance_logits_stack.append(tmp)
        labels_stack.append(np.sum(one_hots[video_label], axis=0))
        tcam = tcam.cpu().data.numpy()
        pred_label = np.argmax(tcam, axis=-1)
        assert len(pred_label) == len(frame_label)
        for gt in frame_label:
            for g in gt:
                res[g - 1] += 1

        score = np.zeros((len(tcam), 1))
        if use_anchor:
            att = att_logits[0].cpu().data.numpy()
            score = att.squeeze()
        segments, frames = generate_segment(
            tcam, score, threshold_type=threshold_type, frame_type=frame_type, act_weight=act_weight, adjust_mean=adjust_mean)
        fps = gt_info[vname].get('fps', fps)
        for frame in frames:
            aps[frame[0]] += 1
            if frame[0] in frame_label[frame[1]]:
                tps[frame[0]] += 1
            rs2[vname] += [{
                'score': float(frame[2] / 100.0),
                'label': classlist[frame[0]].decode('utf-8'),
                'frame': float(frame[1] * stride / fps)
            }]

        for seg in segments:
            rs[vname] += [{
                'score':
                float(seg[3] / 100.0),
                'label':
                str(classlist[seg[0]].decode('utf-8')),
                'segment':
              [float(seg[1] * stride / fps),
                    float(seg[2] * stride / fps)]
            }]
    save_dict['results'] = rs
    frame_dict['results'] = rs2

    with open(prediction_filename, 'w') as fw:
        json.dump(save_dict, fw)
    frame_detection = FrameDetection(
        groundtruth_filename, frame_dict, subset=subset, verbose=True, check_status=False)
    frame_detection.evaluate()
    anet_detection = ANETdetection(groundtruth_filename,
                                   save_dict,
                                   subset=subset,
                                   tiou_thresholds=tiou_thresholds,
                                   verbose=True,
                                   check_status=False)
    dmap = anet_detection.evaluate()
    for i in range(len(dmap)):
        logger.log_value('mAP/IoU@%s' % (str(tiou_thresholds[i])), dmap[i],
                         itr)
    labels_stack = np.array(labels_stack)
    instance_logits_stack = np.array(instance_logits_stack)
    cmap = cmAP(instance_logits_stack, labels_stack)
    print(cmap)
    tp = np.sum(tps)
    ap = np.sum(aps)
    recall = np.sum(res)
    print(
        'All act frames %d, predict all frames : %d, right frames: %d,  AP: %0.5f, Recall: %0.5f'
        % (recall, ap, tp, tp / ap, tp / recall))
    acc = dmap[-3]
    return np.mean(acc)
