import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Block(torch.nn.Module):

    def __init__(self,
                 n_features,
                 dims,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 use_bn=False,
                 dim_reduction=False):
        super(Block, self).__init__()
        cur_dim = n_features
        layers = []
        if dim_reduction:
            layers.append(self._make_conv_layer(n_features, n_features // 2))
            cur_dim = cur_dim // 2
        for dim in dims:
            output_dim = dim
            layers.append(
                self._make_conv_layer(cur_dim,
                                      output_dim,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      stride=stride,
                                      padding=padding,
                                      use_bn=use_bn))
            cur_dim = output_dim
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    def _make_conv_layer(self,
                         input_dim,
                         output_dim,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         dilation=1,
                         use_bn=False):
        module_list = [
            nn.Conv1d(input_dim,
                      output_dim,
                      kernel_size,
                      padding=padding,
                      dilation=dilation)
        ]
        if use_bn:
            module_list += [nn.BatchNorm1d(output_dim)]
        module_list += [nn.ReLU()]
        return nn.Sequential(*module_list)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SFNET(torch.nn.Module):

    def __init__(self, n_feature, n_class, labels101to20=None, use_conv=False):
        super(SFNET, self).__init__()
        self.labels20 = labels101to20
        self.n_class = n_class
        self.use_conv = use_conv
        in_dim = int(n_feature / 2)
        # FC layers for the 2 streams
        out_dim = in_dim
        if use_conv:
            dims = [512, 256]
            self.flow_cls_layer = Block(in_dim,
                                        dims,
                                        kernel_size=3,
                                        dilation=3,
                                        padding=3,
                                        dim_reduction=True)
            self.rgb_cls_layer = Block(in_dim,
                                       dims,
                                       kernel_size=3,
                                       dilation=3,
                                       padding=3,
                                       dim_reduction=True)
            self.flow_act_layer = Block(in_dim,
                                        dims,
                                        kernel_size=7,
                                        dilation=3,
                                        padding=9,
                                        dim_reduction=True)
            self.rgb_act_layer = Block(in_dim,
                                       dims,
                                       kernel_size=7,
                                       dilation=3,
                                       padding=9,
                                       dim_reduction=True)
            out_dim = dims[-1]
        else:
            self.flow_cls_layer = nn.Sequential(
                *[self._make_linear_layer(in_dim, in_dim) for _ in range(2)])
            self.rgb_cls_layer = nn.Sequential(
                *[self._make_linear_layer(in_dim, in_dim) for _ in range(2)])
            self.flow_act_layer = Block(in_dim, [in_dim, in_dim])
            self.rgb_act_layer = Block(in_dim, [in_dim, in_dim])

        self.flow_classifier = nn.Linear(out_dim, n_class)
        self.rgb_classifier = nn.Linear(out_dim, n_class)
        self.flow_action = nn.Linear(out_dim, 1)
        self.rgb_action = nn.Linear(out_dim, 1)

        self.apply(weights_init)
        # Params for multipliers of TCams for the 2 streams
        self.mul_r = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.act_f = nn.Parameter(data=torch.Tensor(1).float().fill_(0))
        self.act_r = nn.Parameter(data=torch.Tensor(1).float().fill_(0))
        self.dropout_f = nn.Dropout(0.7)
        self.dropout_r = nn.Dropout(0.7)
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=1)

    def _make_linear_layer(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def forward(self, inputs, device='cpu', is_training=True, seq_len=None):
        # inputs - batch x seq_len x featSize
        base_x_f = inputs[:, :, 1024:]
        base_x_r = inputs[:, :, :1024]

        if self.use_conv:
            base_x_f = base_x_f.permute([0, 2, 1])
            base_x_r = base_x_r.permute([0, 2, 1])
            c_x_f = self.flow_cls_layer(base_x_f).permute([0, 2, 1])
            c_x_r = self.rgb_cls_layer(base_x_r).permute([0, 2, 1])
            a_x_f = self.flow_act_layer(base_x_f).permute([0, 2, 1])
            a_x_r = self.rgb_act_layer(base_x_f).permute([0, 2, 1])
        else:
            # classification_module
            c_x_f = self.flow_cls_layer(base_x_f)
            c_x_r = self.rgb_cls_layer(base_x_r)

            # proposal module
            a_x_f = self.flow_act_layer(base_x_f.permute([0, 2,
                                                          1])).permute([0, 2, 1])
            a_x_r = self.rgb_act_layer(
                base_x_f.permute([0, 2, 1])).permute([0, 2, 1])

        a_x_f = self.flow_action(a_x_f)
        a_x_r = self.rgb_action(a_x_r)
        a_x = self.act_f * a_x_f + self.act_r * a_x_r
        if is_training:
            c_x_f = self.dropout_f(c_x_f)
            c_x_r = self.dropout_r(c_x_r)
        cls_x_f = self.flow_classifier(c_x_f)
        cls_x_r = self.rgb_classifier(c_x_r)
        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        return c_x_f, cls_x_f, c_x_r, cls_x_r, tcam, a_x_f, a_x_r, a_x
