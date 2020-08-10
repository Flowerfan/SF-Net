import argparse

parser = argparse.ArgumentParser(description='3C-Net')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='number of instances in a batch of data (default: 32)')
parser.add_argument('--model-name', default='sfnet', help='name to save model')
parser.add_argument('--mode', default='single',
                    choices=['single', 'weakly', 'fully'], help='traininig mode')
parser.add_argument('--log-dir', default='./logs',
                    help='diretory for saving logs')
parser.add_argument('--model-dir', default='./ckpt',
                    help='diretory for saving models')
parser.add_argument('--feature-path', default='./data', help='feature path')
parser.add_argument('--resume', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, type=int,
                    help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int, default=101,
                    help='number of classes (default: 101)')
parser.add_argument('--dataset-name', default='Thumos14',
                    help='dataset to train on (default: Thumos14)')
parser.add_argument('--actionness-weight', default=2.0, type=float,
                    help='actionness module for gnerating action proposal')
parser.add_argument('--adjust-mean', default=1, type=float,
                    help='mean parameter for generating proposal')
parser.add_argument('--max-seqlen', type=int, default=750,
                    help='maximum sequence length during training (default: 750)')
parser.add_argument('--alpha', type=float, default=1,
                    help='alpha hyper-parameter')
parser.add_argument('--beta', type=float, default=1,
                    help='beta hyper-parameter')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='gamma hyper-parameter')
parser.add_argument('--max-grad-norm', type=float, default=10,
                    help='value loss coefficient (default: 10)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=8000,
                    help='maximum iteration to train (default: 50000)')
parser.add_argument('--eval-steps', type=int, default=1000, help='video fps')
parser.add_argument('--tm', type=float, default=7, help='background ratio')
parser.add_argument('--fps', type=int, default=25, help='video fps')
parser.add_argument('--stride', type=int, default=16, help='feature stride')
parser.add_argument('--summary', default='no summary', help='Summary of expt')
parser.add_argument('--frame-type', default='max',
                    type=str, help='frame generation method')
parser.add_argument('--activity-net', action='store_true',
                    default=False, help='ActivityNet v1.2 dataset')
parser.add_argument('--eval-only', action='store_true',
                    default=False, help='Evaluation only performed')
parser.add_argument('--use-sf', action='store_false',
                    help='using human annotations, defautl true')
parser.add_argument('--background', action='store_false',
                    help='use background or not, default true')
parser.add_argument('--expand', action='store_false',
                    help='expand action frames, default true')
parser.add_argument('--expand-step', default=2000, type=int,    
                    help='step to start expanding frame')
parser.add_argument('--use-anchor', action='store_false',
                    help='use actoiness score or not, default true')
