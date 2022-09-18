import torch
import torch.nn as nn
from torch.nn import functional as F

class NonLocalBlock(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, mode='embedded', act='relu', norm='layer'):
        '''

        :param conv: convolutional layer e.g., nn.Conv1d, nn.Conv2d
        :param filters_in: in_channels
        :param filters: out_channels
        :param height:
        :param stride: convolution stride
        :param mode:
        :param act:
        :param norm:
        :return:
        '''
        super(NonLocalBlock, self).__init__()

        self.mode = mode
        self.filters = filters

        # init convolutional layer for g(x)
        self.conv1 = nn.Sequential(
            conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
        )

        # init the decoder
        if norm == 'layer':
            self.W_z = nn.Sequential(
                conv(filters, filters_in, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm((filters_in, height)) # changed this
            )
        elif norm == 'batch':
            self.W_z = nn.Sequential(
                conv(filters, filters_in, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.BatchNorm2d(filters_in)
            )

        # nn.init.constant_(self.W_z[1].weight, 0)
        # nn.init.constant_(self.W_z[1].bias, 0)

        # init the components for the pairwise function
        if self.mode == 'embedded' or self.mode == "concatenate":
            self.phi = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)
            self.theta = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)

        if self.mode == "concatenate":
            if act == 'relu':
                self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.filters * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            elif act == 'swish':
                self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.filters * 2, out_channels=1, kernel_size=1),
                    nn.SiLU(inplace=True)
                )

    def forward(self, x):

        # x is of dimension [batch_size, spatial resolution, hidden dim]
        batch_size = x.size(0)

        # g is of dim [batch_size, filters, spatial_dim]
        g = self.conv1(x).view(batch_size, self.filters, -1)
        g = g.permute(0, 2, 1) # change g to shape [batch_size, spatial res, filters]

        # compute the pairwise function
        if self.mode == 'embedded':
            theta_x = self.theta(x).view(batch_size, self.filters, -1) # theta and phi map to [batch_size, filters, spatial res]
            phi_x = self.phi(x).view(batch_size, self.filters, -1)
            theta_x = theta_x.permute(0, 2, 1) # transform theta to [batch_size, spatial res, filters]
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.filters, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.filters, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        # calculate the normalization factor and normalize
        if self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N

        # compute the output at each position
        out = torch.matmul(f_div_C, g)

        # contiguous here just allocates contiguous chunk of memory
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.filters, *x.size()[2:])

        # decode
        out = self.W_z(out)

        # residual connection
        out = out + x

        return out