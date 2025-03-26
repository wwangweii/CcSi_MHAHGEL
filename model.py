import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from typing import Any, Optional, Tuple
from torch.autograd import Function
from sklearn.metrics.pairwise import cosine_similarity




class globalavg(nn.Module):
    def __init__(self, **kwargs):
        super(globalavg, self).__init__(**kwargs)

    def forward(self, x, dim=1):
        # return torch.mean(x, dim=1, keepdim=True)
        return torch.mean(x, dim=dim)


class globalmax(nn.Module):
    def __init__(self, **kwargs):
        super(globalmax, self).__init__(**kwargs)

    def forward(self, x, dim=1):
        # return torch.mean(x, dim=1, keepdim=True)
        x, _ = torch.max(x, dim=dim)
        return x



class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.cuda.FloatTensor(in_ft, out_ft), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, H, W):
        DV2_H, invDE_HT_DV2 = self.generate(H)
        x = torch.matmul(x, self.weight)  # node feature transfomer
        x = torch.matmul(invDE_HT_DV2, x)
        temp = torch.diag(torch.ones(H.shape[1]))
        temp = temp.to(x.device)
        diag_W = torch.mul(W.unsqueeze(1), temp)
        x = torch.matmul(diag_W, x)
        x = torch.matmul(DV2_H, x)
        if self.bias is not None:
            x = x + self.bias
        return x

    def generate(self, H):
        DV = torch.sum(H, axis=1)
        DE = torch.sum(H, axis=0)
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -0.5))
        HT = H.T
        DV2_H = torch.mm(DV2, H)
        invDE_HT_DV2 = torch.mm(torch.mm(invDE, HT), DV2)
        return DV2_H, invDE_HT_DV2


class CcSi_MHAHGEL(nn.Module):
    def __init__(self, node1: int, mid1: int, out1: int, node2: int, mid2: int, out2: int, k, dropout=0.):
        super(CcSi_MHAHGEL, self).__init__()
        self.gap = globalavg()
        self.gmp = globalmax()
        self.dropout = dropout
        self.proj11 = Parameter(torch.cuda.FloatTensor(node1, 1), requires_grad=True)
        self.proj21 = Parameter(torch.cuda.FloatTensor(node2, 1), requires_grad=True)
        self.proj12 = Parameter(torch.cuda.FloatTensor(mid1, 1), requires_grad=True)
        self.proj22 = Parameter(torch.cuda.FloatTensor(mid2, 1), requires_grad=True)
        self.hgc11 = HGNN_conv(node1, mid1)
        self.hgc12 = HGNN_conv(mid1, out1)
        self.hgc21 = HGNN_conv(node2, mid2)
        self.hgc22 = HGNN_conv(mid2, out2)
        self.k = k
        self.embedding = nn.Sequential(
            nn.Linear((out1 + out2) * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.proj11.size(0))
        self.proj11.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.proj21.size(0))
        self.proj21.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.proj12.size(0))
        self.proj12.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.proj22.size(0))
        self.proj22.data.uniform_(-stdv, stdv)
    def updateW(self, H, x, proj):
        x = F.dropout(x, p=0.3, training=self.training)
        edge_feature = torch.matmul(x.permute(0, 2, 1), H)
        score = torch.matmul(edge_feature.permute(0, 2, 1), proj).squeeze(2)
        score = self.sigmoid(score)
        return score
    
    def forward(self, x1, x2, H1, H2):
        W1 = self.updateW(H1, x1, self.proj11)
        x1 = F.relu(self.hgc11(x1, H1, W1))
        W1 = self.updateW(H1, x1, self.proj12)
        x1 = F.relu(self.hgc12(x1, H1, W1))
        x1 = torch.cat([self.gap(x1), self.gmp(x1)], dim=1)
        W2 = self.updateW(H2, x2, self.proj21)
        x2 = F.relu(self.hgc21(x2, H2, W2))
        W2 = self.updateW(H2, x2, self.proj22)
        x2 = F.relu(self.hgc22(x2, H2, W2))
        x2 = torch.cat([self.gap(x2), self.gmp(x2)], dim=1)
        x = torch.cat([(1-self.k)*x1, self.k*x2], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(x.shape[0], -1)
        embedding = x
        x = self.embedding(x)
        return x, embedding

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 在计算参数w的初值时考虑到输出该层元素的个数n
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # 全连通乘积
        support = support.cuda()
        output = torch.matmul(adj, support)  # 添加掩模
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Cat(nn.Module):
    def __init__(self, **kwargs):
        super(Cat, self).__init__(**kwargs)

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


def Eu_dis(x, metric='euclidean'):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    d = scipy.spatial.distance.pdist(x, metric)
    # 用于计算样本对之间的欧式距离
    d = scipy.spatial.distance.squareform(d)
    # 将样本间距离用方阵表示出来
    return d


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge

    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    W = np.mat(np.diag(W))
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(feature, k_neig, is_probH = True, normalize = False):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    dis_mat = Eu_dis(feature)
    # dis_mat = feature
    # 用于计算样本对之间的欧式距离
    # 将样本间距离用方阵表示出来
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    # n_obj centroid edge and K cluster edge
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx, :]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / 2 * avg_dis ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    if normalize:
        for i in range(np.size(H,1)):
            sum = np.sum(H[:, i])
            H[:, i] = H[:, i]/sum
    return H

def construct_H_with_KNN_from_cosine(feature, k_neig, is_probH = True, normalize = False):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    cor_mat = cosine_similarity(feature)
    #print(cor_mat)
    # dis_mat = feature
    # 用于计算样本对之间的欧式距离
    # 将样本间距离用方阵表示出来
    n_obj = cor_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    # n_obj centroid edge and K cluster edge
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        cor_mat[center_idx, center_idx] = 1
        cor_vec = cor_mat[center_idx, :]
        idx = np.array(np.argsort(-cor_vec)).squeeze()
        if not np.any(idx[:k_neig] == center_idx):
            idx[k_neig - 1] = center_idx
        for node_idx in idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = cor_vec[node_idx]
            else:
                H[node_idx, center_idx] = 1.0
    if normalize:
        for i in range(np.size(H,1)):
            sum = np.sum(H[:, i])
            H[:, i] = H[:, i]/sum
    return H

def construct_H_with_KNN_from_pearson(feature, k_neig, is_probH=True):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    cor_mat = np.corrcoef(feature)
    #print(cor_mat)
    # dis_mat = feature
    # 用于计算样本对之间的欧式距离
    # 将样本间距离用方阵表示出来
    n_obj = cor_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    # n_obj centroid edge and K cluster edge
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        cor_mat[center_idx, center_idx] = 1
        cor_vec = cor_mat[center_idx, :]
        idx = np.array(np.argsort(-abs(cor_vec))).squeeze()
        if not np.any(idx[:k_neig] == center_idx):
            idx[k_neig - 1] = center_idx
        for node_idx in idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = abs(cor_vec[node_idx])
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN_from_cluster(feature, K=10, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    kmeans = KMeans(n_clusters=K).fit(feature)
    label = kmeans.labels_
    centroid = kmeans.cluster_centers_
    # 用于计算样本对之间的欧式距离
    # 将样本间距离用方阵表示出来
    n_obj = feature.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = K
    # n_obj centroid edge and K cluster edge
    H = np.zeros((n_obj, n_edge))
    for i in range(K):
        cluster = feature[label == i, :]
        centroid_i = centroid[pairwise_distances_argmin(cluster, centroid, metric="euclidean")[0]].reshape(1, -1)
        d = pairwise_distances(cluster, centroid_i)[:, 0]
        avg_dis = np.average(d) + 1e-5
        if is_probH:
            H[label == i, i] = np.exp(-d ** 2 / (m_prob * avg_dis) ** 2)[:]
        else:
            H[label == i, i] = 1
    return H

