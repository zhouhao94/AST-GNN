import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
import networkx as nx


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, norm_lap_matr=False, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # leakyrelu parameter
        self.concat = concat  # if true, elu activation function
        self.norm_lap_matr = norm_lap_matr        

        # parameter W and a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # W
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # a
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, inp, adj):
        """
        :param inp: input_feature [N, in_features]
        :param adj: adjacent matrix [N, N]
        :return:
        """
        n, t, v, c = inp.size()
        N = v
        h = torch.einsum('ntvc,co->ntvo', (inp, self.W))

        a_input = torch.cat([h.repeat(1, 1, 1, N).view(n, t, N * N, -1),
                        h.repeat(1, 1, N, 1)], dim=1).view(n, t, N, -1, 2 * self.out_features)  # (n, t, N, N, 2*outputs)
        e = self.leakyrelu(torch.einsum('ntvwo,of->ntvwf', (a_input, self.a))).squeeze(4)  # (n, t, N, N)

        zero_vec = -1e12 * torch.ones_like(e)
        adj = adj.unsqueeze(0).repeat(n, 1, 1, 1)  # (t, v, v) -> (1, t, N, N) -> (n, t, N, N)
        if adj.size()[1] != t:
            attention = e
        else:
            # attention = torch.where(adj > 0, e, zero_vec)
            attention = e
        attention = Func.softmax(attention, dim=3)
        if self.norm_lap_matr == True:
            norm_att = attention
            norm_att = norm_att.data.cpu().numpy()
            for i in range(attention.size()[0]):
                for j in range(attention.size()[1]):
                    G = nx.from_numpy_matrix(norm_att[i,j,:,:])
                    norm_att[i,j,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            attention = torch.from_numpy(norm_att).cuda()
        attention = Func.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('ntvw,ntwh->ntvh', (attention, h))  # (n, t, N, o)
        if self.concat:
            return Func.elu(h_prime), attention
        else:
            return h_prime, attention


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.bn =  nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        x = self.bn(x)
        x = self.prelu(x)
        return x.contiguous()


class TGNN(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(TGNN,self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1),
                dilation=(t_dilation, 1),
                bias=bias),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                out_channels,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1),
                dilation=(t_dilation, 1),
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        self.GAT = GraphAttentionLayer(inter_channels, inter_channels, 0.6, 0.2, norm_lap_matr=False, concat=True)

    def forward(self, x, A):
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
        """
        assert A.size(0) == self.kernel_size
        x = self.conv1(x)
        # x = x.permute(0, 2, 3, 1)  # n c t v -> n t v c
        # x = self.GAT(x, A).permute(0, 3, 1, 2)  # n t v c -> n c t v
        x = x.permute(0, 3, 2, 1)  # n c t v -> n v t c
        x, A_t = self.GAT(x, A)
        x = self.conv2(x.permute(0, 3, 2, 1).contiguous()) # n, v, t, c -> n, c, t, v
        return x.contiguous(), A_t


class SGNN(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(SGNN,self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1),
                dilation=(t_dilation, 1),
                bias=bias),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                out_channels,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1),
                dilation=(t_dilation, 1),
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        self.GAT = GraphAttentionLayer(inter_channels, inter_channels, 0.6, 0.2, norm_lap_matr=False, concat=True)

    def forward(self, x, A):
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
        """
        assert A.size(0) == self.kernel_size
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # n c t v -> n t v c
        x, A_s = self.GAT(x, A) 
        x = self.conv2(x.permute(0, 3, 1, 2).contiguous()) # n t v c -> n c t v
        return x.contiguous(), A_s


class st_gnn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gnn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = SGNN(in_channels, inter_channels, out_channels, kernel_size[1])
        self.tcn = TGNN(out_channels, inter_channels, out_channels, kernel_size[1])

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A_s = self.gcn(x, A)
        x, A_t = self.tcn(x, A)
        x = x + res
        
        if not self.use_mdn:
            x = self.prelu(x)
        #_ = 0
        return x,A_s,A_t

class ast_gnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,inter_feat=32, output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(ast_gnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gnn(input_feat, inter_feat, output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gnn(output_feat, inter_feat, output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,(3, 1),padding=(1, 0)))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,(3, 1),padding=(1, 0))
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


        
    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a_s,a_t = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        
        return v,a_s,a_t
