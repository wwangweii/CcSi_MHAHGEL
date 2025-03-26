import math
import numpy as np
import scipy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
from torch.autograd import Variable
from tool import *
from model import*
from prepare_data import load_data
from sklearn.model_selection import StratifiedKFold
import random
import os
import warnings
import argparse
import json
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from datetime import date
import os
current_date = date.today()

import matplotlib.pyplot as plt
from itertools import repeat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(100)


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--out', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--neigs', type=int, default=3, help='Neighbours for KNN.')
parser.add_argument('--batchsize', type=int, default=24, help='Batch size for training.')
parser.add_argument('--k', type=float, default=0.7, help='Weighted concatenate.')
parser.add_argument('--alpha', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--beta', type=float, default=0, help='Regularization for class-consistency loss.')
parser.add_argument('--thres', type=float, default=0.5, help='Thres for FCs.')
parser.add_argument('--theta', type=float, default=0, help='Regularization for site-consistency loss.')
parser.add_argument('--atlas', nargs='+', type=str, default=['aal','ho'], help='Atlas types.')
parser.add_argument('--margin', type=float, default=4, help='Margin for intra-class loss.')
parser.add_argument('--site_num', type=int, default=4, help='The number of sites.')
parser.add_argument('--drop_last', type=bool, default=True, help='Drop last for Dataloader.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class MyDataset(Dataset):
    def __init__(self, data1, data2, multi_label):
        self.len = np.size(data1, 0)
        self.data1 = data1
        self.data2 = data2
        self.multi_label = multi_label
    def __getitem__(self, index):
        return self.data1[index, :], self.data2[index, :], self.multi_label[index, 0], self.multi_label[index, 1], self.multi_label[index, 2]
    def __len__(self):
        return self.len



def get_population_sim(target, site):
    target = target.cpu().detach().numpy()
    site = site.cpu().detach().numpy()
    n_obj = target.shape[0]
    Sim1 = np.zeros((n_obj, n_obj))
    Sim2 = np.zeros((n_obj, n_obj))
    for i in range(n_obj):
        for j in range(n_obj):
            if target[i] == target[j]:
                Sim1[i, j] = 1
            else:
                Sim1[i, j] = -1

            if site[i] != site[j] and target[i] == target[j]:
                Sim2[i, j] = 1
            else:
                Sim2[i, j] = 0
    return Sim1, Sim2



def get_population_loss(data, Sim, m=args.margin, avg=True):
    m = torch.tensor(m)
    loss_intra = torch.tensor(0).type(torch.FloatTensor).cuda()
    loss_inter = torch.tensor(0).type(torch.FloatTensor).cuda()
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i >= j:
                if Sim[i, j] > 0:
                    loss_intra = loss_intra + Sim[i, j] * torch.norm(data[i, :]-data[j, :], 2)
                if Sim[i, j] < 0:
                    loss_inter = loss_inter - Sim[i, j] * torch.max(torch.tensor(0), m - torch.norm(data[i, :]-data[j, :], 2))
    if avg:
        loss_intra = loss_intra/data.shape[0]
        loss_inter = loss_inter/data.shape[0]
    loss = loss_intra + loss_inter
    return loss, loss_intra, loss_inter


# def get_population_loss(data, Sim, m=args.margin, avg=True):
#     m = torch.tensor(m, device=data.device)
#     norms = torch.norm(data[:, None] - data, p=2, dim=2)  # Calculate norms for all pairs
#     Sim = Sim * (norms > 0)  # Only consider positive similarities

#     loss_intra = torch.sum(Sim * norms)  # Sum intra-cluster loss
#     loss_inter = torch.sum(Sim * torch.clamp(m - norms, min=0))  # Sum inter-cluster loss

#     if avg:
#         num_pairs = torch.sum(Sim > 0)  # Count number of positive similarity pairs
#         loss_intra = loss_intra / num_pairs if num_pairs > 0 else loss_intra
#         loss_inter = loss_inter / num_pairs if num_pairs > 0 else loss_inter

#     loss = loss_intra + loss_inter
#     return loss, loss_intra, loss_inter


def one_hot(site, site_num = args.site_num):
    sub = site.shape[0]
    D = torch.zeros((sub, site_num))
    for i in range(sub):
        D[i, site[i].long()] = 1
    if args.cuda:
        D = D.cuda()
    return D

def centering(X):
    sub = X.shape[0]
    H = torch.eye(sub) - (1 / sub) * torch.ones((sub, sub))
    if args.cuda:
        H = H.cuda()
    return torch.mm(X, H)

def rbf(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    KX = torch.exp(KX)
    return KX

def HSIC(data, site, site_num=args.site_num, kernel="linear"):
    sub = data.shape[0]
    D = one_hot(site, site_num)
    if kernel == "linear":
        K = torch.mm(data, data.T)
    elif kernel == "rbf":
        K = rbf(data)
    L = torch.mm(D, D.T)
    loss = torch.trace(torch.mm(centering(K), centering(L)))/((sub-1)**2)
    return loss

def train(data1, data2, H1, H2, target):
    batch_size = args.batchsize
    base_lr = args.lr
    maxepoch = args.epochs
    alpha = args.alpha
    beta = args.beta
    theta = args.theta
    variable_weight = True
    out = args.out
    hidden = args.hidden
    dataset = MyDataset(data1, data2, target)
    label = target[:, 0]
    site = target[:, 1]
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=args.drop_last)
    
    torch.cuda.empty_cache()
    classifier = CcSi_MHAHGEL(np.size(data1, 1), hidden, out, np.size(data2, 1), hidden, out, k=args.k, dropout=args.dropout)
    if args.cuda:
        classifier = classifier.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=base_lr, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    starttime = time.time()
    
    for epoch in range(maxepoch):
        loss_history = []
        classifier.train()
        
        for x1, x2, y1, y2, y3 in data_loader:
            x1, x2, y1, y2 = Variable(x1).type(torch.FloatTensor), Variable(x2).type(torch.FloatTensor), Variable(y1).type(torch.FloatTensor), Variable(y2).type(torch.FloatTensor)
            
            if args.cuda:
                x1, x2, y1, y2 = x1.cuda(), x2.cuda(), y1.cuda(), y2.cuda()
            
            result, embedding = classifier(x1, x2, H1, H2)
            l2_regularization = torch.tensor(0.).to(x1.device)
            W_loss = torch.tensor(0.).to(x1.device)
            Sim1, _ = get_population_sim(y1, y2)
            HSIC_loss = HSIC(embedding, y2, kernel="linear")
            population_loss, intra, inter = get_population_loss(embedding, Sim1)
            
            for name, param in classifier.named_parameters():
                if variable_weight and (name == 'W1' or name == 'W2'):
                    W_loss += torch.norm(param, 2)
                elif "bias" not in name:
                    l2_regularization += torch.norm(param, 2)
            
            loss1 = criterion(result, y1.long())
            loss = loss1 + alpha * l2_regularization + beta * population_loss + theta * HSIC_loss
            loss_history.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            print("population_loss: ", population_loss)
            print("intra_loss: ", intra)
            print("inter_loss: ", inter)
            print("Classification Loss: ", loss1)
            print("HSIC loss: ", HSIC_loss)
            print("L2 Loss: ", l2_regularization)
        
        with torch.no_grad():
            pred, _ = predict(classifier, data1, data2, H1, H2)
            acc = calulate_acc(pred, label)
        
        print('Epoch :', epoch, '|', 'train_loss:%.5f' % np.mean(loss_history), '|',
              'train_acc:%.4f' % acc)
    
    endtime = time.time()
    print('训练耗时：', (endtime - starttime))
    return classifier

def predict(model, data1, data2, H1, H2):
    data1 = torch.Tensor(data1).type(torch.FloatTensor)
    data2 = torch.Tensor(data2).type(torch.FloatTensor)
    if args.cuda:
        data1 = data1.cuda()
        data2 = data2.cuda()
        model = model.cuda()
    model.eval()
    output, _ = model.forward(data1, data2, H1, H2)
    output = output.view(-1, 2)
    pred = torch.argmax(output, dim=1).detach().cpu().numpy()
    prob = F.softmax(output, dim=1)
    prob = prob[:, 1].detach().cpu().numpy()
    return pred, prob

ROI1 = 116
ROI2 = 200
ROI3 = 111

if __name__ == '__main__':
    selected_atlas = args.atlas
    feature1, feature2, information = load_data(*selected_atlas)
    sub = np.size(feature1, 0)
    print("The total subject: ", sub)
    print(information.head(0))
    site_name = ['NYU', 'UM_1', 'USM', 'YALE']
    seleted_site, seleted_index = seleted_site_by_name(feature1, information, site_name)
    # threshold = 20
    # seleted_site, seleted_index = seleted_site_by_num(feature1, information, 20)
    print("Site: ", seleted_site)
    feature1 = feature1[seleted_index, :]
    feature2 = feature2[seleted_index, :]
    ROI1 = int(np.sqrt(np.size(feature1, 1)))
    ROI2 = int(np.sqrt(np.size(feature2, 1)))
    print("The number of ROIs on the atlas1: ", ROI1)
    print("The number of ROIs on the atlas2: ", ROI2)

    information = information.iloc[seleted_index]
    information.reset_index()
    site_label = np.array(information['site'])
    sex_label = np.array(information['sex'])
    female = np.sum(sex_label - 1)
    male = len(sex_label) - female
    print("Male: ", male)
    print("Female: ", female)
    print(sex_label)
    site = np.unique(site_label)
    print("All Site: ", site)
    sub = np.size(feature1, 0)
    print("The seleted subject: ", sub)
    site_target = np.zeros(sub)
    for i, name in enumerate(site):
        site_target[site_label == name] = int(i)
    data1 = np.zeros((sub, ROI1, ROI1))
    data2 = np.zeros((sub, ROI2, ROI2))

    site_label = np.array(information['site'])
    site = np.unique(site_label)
    target = np.array(information['diagnosis'])
    target = 2 - target[:]
    label = np.array(target, dtype=int)
    # assume asd as 1 and tdc as 0
    num_asd = np.sum(target)
    num_tdc = sub - num_asd
    print("The num of ASD: ", num_asd)
    print("The num of TDC: ", num_tdc)
    multi_label = np.zeros((sub, 3))
    multi_label[:, 0] = label    #asd/tdc
    multi_label[:, 1] = site_target #site
    multi_label[:, 2] = sex_label  # site
    for i in range(sub):
        temp1 = np.reshape(feature1[i, :], (ROI1, ROI1))
        temp2 = np.reshape(feature2[i, :], (ROI2, ROI2))
        data1[i, :, :] = 1 / 2 * np.log((1 + temp1) / (1 - temp1))
        data2[i, :, :] = 1 / 2 * np.log((1 + temp2) / (1 - temp2))
    best_each_accuracy = []
    best_each_sensitivity = []
    best_each_precision = []
    best_each_F1scores = []
    best_acc = 0
    best_alpha = 0
    best_theta = 0
    state = [0]
    all_state_accuracy = []
    all_state_sensitivity = []
    all_state_precision = []
    all_state_F1scores = []
    all_state_AUC = []
    for state_i in state:
        iter = 0
        test_each_accuracy = []
        test_each_sensitivity = []
        test_each_precision = []
        test_each_F1scores = []
        test_each_AUC = []
        for X_train_i, X_test_i in inter_CV(data1, multi_label, information, k=10, state=state_i):
            iter = iter + 1
            X_train1, X_test1 = data1[X_train_i, :, :], data1[X_test_i, :, :]
            X_train2, X_test2 = data2[X_train_i, :, :], data2[X_test_i, :, :]
            Y_train, Y_test = multi_label[X_train_i, :], multi_label[X_test_i]
            print(np.size(X_train1, axis=0))
            print(np.size(X_test1, axis=0))
            diagnosis = Y_train[:, 0]
            K_neigs1 = args.neigs
            K_neigs2 = args.neigs
            average_FC11 = np.mean(X_train1[diagnosis == 1, :], axis=0)
            average_FC12 = np.mean(X_train1[diagnosis == 0, :], axis=0)
            average_FC21 = np.mean(X_train2[diagnosis == 1, :], axis=0)
            average_FC22 = np.mean(X_train2[diagnosis == 0, :], axis=0)
            H11 = construct_H_with_KNN_from_distance(average_FC11, K_neigs1, True)
            H12 = construct_H_with_KNN_from_distance(average_FC12, K_neigs1, True)
            H21 = construct_H_with_KNN_from_distance(average_FC21, K_neigs2, True)
            H22 = construct_H_with_KNN_from_distance(average_FC22, K_neigs2, True)
            # average_FC11[average_FC11 <= args.thres] = 0
            # average_FC12[average_FC12 <= args.thres] = 0
            # average_FC21[average_FC21 <= args.thres] = 0
            # average_FC22[average_FC22 <= args.thres] = 0
            # H11 = average_FC11 + np.eye((average_FC11.shape[0]))
            # H12 = average_FC12 + np.eye((average_FC12.shape[0]))
            # H21 = average_FC21 + np.eye((average_FC21.shape[0]))
            # H22 = average_FC22 + np.eye((average_FC22.shape[0]))
            H1 = hyperedge_concat(H11, H12)
            H2 = hyperedge_concat(H21, H22)
            H1 = torch.tensor(H1).type(torch.FloatTensor)
            H2 = torch.tensor(H2).type(torch.FloatTensor)
            if args.cuda:
                H1 = H1.cuda()
                H2 = H2.cuda()
            print("Start Training the Model")
            model = train(X_train1, X_train2, H1, H2, Y_train)
            model.eval()
            print("Start Predicting")
            pred, prob = predict(model, X_test1, X_test2, H1, H2)
            accuracy, sensitivity, precision, F1_score = evaluate(pred, Y_test[:, 0])
            AUC = get_auc(prob, Y_test[:, 0])
            print("The " + str(iter) + " CV Test Accuracy: ", accuracy)
            test_each_accuracy.append(accuracy)
            test_each_sensitivity.append(sensitivity)
            test_each_precision.append(precision)
            test_each_F1scores.append(F1_score)
            test_each_AUC.append(AUC)
            del model
        all_state_accuracy.append(np.mean(test_each_accuracy))
        all_state_sensitivity.append(np.mean(test_each_sensitivity))
        all_state_precision.append(np.mean(test_each_precision))
        all_state_F1scores.append(np.mean(test_each_F1scores))
        all_state_AUC.append(np.mean(test_each_AUC))
        print("State: ", state_i)
        print("All Test Accuracy: ", test_each_accuracy)
        print("The Mean Test Accuracy: ", np.mean(test_each_accuracy))
        print("The Mean Test sensitivity: ", np.mean(test_each_sensitivity))
        print("The Mean Test precision: ", np.mean(test_each_precision))
        print("The Mean Test F1scores: ", np.mean(test_each_F1scores))
        print("The Mean AUC: ", np.mean(test_each_AUC))
    print("All Accuracy: ", all_state_accuracy)
    print("All Sensitivity: ", all_state_sensitivity)
    print("All Precision: ", all_state_precision)
    print("All F1scores: ", all_state_F1scores)
    print("All AUC: ", all_state_AUC)
    print("The Mean Accuracy: ", np.mean(all_state_accuracy), ", Std: ", np.sqrt(np.var(all_state_accuracy)))
    print("The Mean Sensitivity: ", np.mean(all_state_sensitivity), ", Std: ",
          np.sqrt(np.var(all_state_sensitivity)))
    print("The Mean Precision: ", np.mean(all_state_precision), ", Std: ", np.sqrt(np.var(all_state_precision)))
    print("The Mean F1scores: ", np.mean(all_state_F1scores), ", Std: ", np.sqrt(np.var(all_state_F1scores)))
    print("The Mean AUC: ", np.mean(all_state_AUC), ", Std: ", np.sqrt(np.var(all_state_AUC)))
    res = {}
    res['all_state_accuracy'] = all_state_accuracy
    res['all_state_sensitivity'] = all_state_sensitivity
    res['all_state_precision'] = all_state_precision
    res['all_state_F1scores'] = all_state_F1scores
    res['all_state_AUC'] = all_state_AUC
    res['metric'] = [(np.mean(all_state_accuracy), np.sqrt(np.var(all_state_accuracy))), 
                     (np.mean(all_state_precision), np.sqrt(np.var(all_state_precision))),
                     (np.mean(all_state_precision), np.sqrt(np.var(all_state_precision))),
                     (np.mean(all_state_F1scores), np.sqrt(np.var(all_state_F1scores))),
                     (np.mean(all_state_AUC), np.sqrt(np.var(all_state_AUC))),]

    output_file = f"./result/CcSiMAHGEL_beta{args.beta}_theta{args.theta}_{current_date}.json"
    with open(output_file, 'w') as fp:
        json.dump(res, fp)