from __future__ import print_function
import torch.optim as optim
import argparse
import os
import torch
from model import SFNET
from video_dataset import Dataset
from eval import evaluate
from train import *
from tensorboard_logger import Logger
from datetime import datetime
import options
from center_loss import CenterLoss


torch.set_default_tensor_type('torch.cuda.FloatTensor')
args = options.parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    feature_dim = int(args.feature_size / 2)
    tiou_thresholds = np.linspace(0.1, 0.7, 7)
    train_subset = 'training'
    test_subset = 'validation'
    threshold_type = 'mean'
    prediction_filename = './data/prediction.json'
    fps = 25
    stride = 16
    if args.dataset_name == 'Thumos14':
        t_max = 750
        t_max_ctc = 2800
        train_subset = 'validation'
        test_subset = 'test'
        num_class = 20
        groundtruth_filename = './data/th14_groundtruth.json'
    elif args.dataset_name == 'GTEA':
        fps = 15
        t_max = 100
        t_max_ctc = 150
        num_class = 7
        groundtruth_filename = './data/gtea_groundtruth.json'
    elif args.dataset_name == 'BEOID':
        fps = 30
        t_max = 100
        t_max_ctc = 400
        num_class = 34
        groundtruth_filename = './data/beoid_groundtruth.json'
    else:
        raise ValueError('wrong dataset')

    device = torch.device("cuda")
    if args.background:
        num_class += 1

    dataset = Dataset(args,
                      groundtruth_filename,
                      train_subset=train_subset,
                      test_subset=test_subset,
                      mode=args.mode,
                      use_sf=args.use_sf)

    os.system('mkdir -p %s' % args.model_dir)
    os.system('mkdir -p %s/%s' % (args.log_dir, args.model_name))
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    logger = Logger('%s/%s_%s' % (args.log_dir, args.model_name, dt_string))

    model = SFNET(dataset.feature_size, num_class).to(device)

    if args.eval_only and args.resume is None:
        print('***************************')
        print('Pretrained Model NOT Loaded')
        print('Evaluating on Random Model')
        print('***************************')

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    criterion_cent_f = CenterLoss(num_classes=num_class,
                                  feat_dim=feature_dim,
                                  use_gpu=True)
    optimizer_centloss_f = torch.optim.SGD(
        criterion_cent_f.parameters(), lr=0.1)
    criterion_cent_r = CenterLoss(num_classes=num_class,
                                  feat_dim=feature_dim,
                                  use_gpu=True)
    optimizer_centloss_r = torch.optim.SGD(
        criterion_cent_r.parameters(), lr=0.1)

    criterion_cent_all = [criterion_cent_f, criterion_cent_r]
    optimizer_centloss_all = [optimizer_centloss_f, optimizer_centloss_r]
    center_f = criterion_cent_f.get_centers()
    center_r = criterion_cent_r.get_centers()
    centers = [center_f, center_r]
    params = {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma}

    ce = torch.nn.CrossEntropyLoss().cuda()
    counts = dataset.get_frame_counts()
    print('total %d annotated frames' % counts)

    for itr in range(args.max_iter + 1):
        dataset.t_max = t_max
        if itr % 2 == 0 and itr > 000:
            dataset.t_max = t_max_ctc
        if not args.eval_only:
            train_SF(itr, dataset, args, model, optimizer, criterion_cent_all,
                     optimizer_centloss_all, logger, device, ce, params, mode=args.mode)

        if itr % args.eval_steps == 0 and (not itr == 0 or args.eval_only):
            print('model_name: %s' % args.model_name)
            acc = evaluate(itr,
                           dataset,
                           model,
                           logger,
                           groundtruth_filename,
                           prediction_filename,
                           background=args.background,
                           fps=fps,
                           stride=stride,
                           subset=test_subset,
                           threshold_type=threshold_type,
                           frame_type=args.frame_type,
                           adjust_mean=args.adjust_mean,
                           act_weight=args.actionness_weight,
                           tiou_thresholds=tiou_thresholds,
                           use_anchor=args.use_anchor)
            torch.save(model.state_dict(),
                       '%s/%s.%d.pkl' % (args.model_dir, args.model_name, itr))
            if acc >= best_acc and not args.eval_only:
                torch.save(model.state_dict(),
                           '%s/%s_best.pkl' % (args.model_dir, args.model_name))
                best_acc = acc
        if args.expand and itr == args.expand_step:
            act_expand(args,
                          dataset,
                          model,
                          device,
                          centers=None)
            model = SFNET(dataset.feature_size, num_class).to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=0.0005)
            counts = dataset.get_frame_counts()
            print('total %d frames' % counts)
        if args.eval_only:
            print('Done Eval!')
            break


if __name__ == '__main__':
    main()
