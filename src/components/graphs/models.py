from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn
import dgl.function as fn
import torch
import torch.nn.functional as F
import math

"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""

class GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GcnSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        
        if not self.use_pp:
            norm = self.get_norm(g)
            g.ndata['h'] = h
            # todo check edge features
            g.update_all(fn.u_mul_e('h', 'feat', 'm'),
                         fn.sum(msg='m', out='h'))
            #g.update_all(fn.copy_u('h', 'm'),
            #             fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GcnSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp=False):
        super(GcnSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(GcnSAGELayer(in_feats, n_hidden, activation=activation,
                                        dropout=dropout, use_pp=use_pp, use_lynorm=True))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(
                GcnSAGELayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
                             use_pp=False, use_lynorm=True))
        # output layer
        self.layers.append(GcnSAGELayer(n_hidden, n_classes, activation=None,
                                        dropout=False, use_pp=False, use_lynorm=False))

    def forward(self, g): #, padding=False):
        h = g.ndata['feat']
            
        # if padding:
        #     max_size = get_in_feats_(None, padding)
        #     if h.shape[1] < max_size:
        #         h = F.pad(h, (0, max_size - h.shape[1]))
                
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(g, h)
        return h

class WeightedMeanSAGELayer(nn.Module):
    """Graph convolution module used by the GraphSAGE model with edge weights.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(WeightedMeanSAGELayer, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h, w):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        w : Tensor
            The edge weight.
        """
        
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)

class MeanSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, n_layers):
        super(MeanSAGE, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(WeightedMeanSAGELayer(in_feats, h_feats))
        for i in range(n_layers - 1):
            self.layers.append(WeightedMeanSAGELayer(h_feats, h_feats))
        self.layers.append(WeightedMeanSAGELayer(h_feats, num_classes))

    def forward(self, g, h, w):
        for l, layer in enumerate(self.layers):
            h = layer(g, h, w)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = F.normalize(h)
        return h
    