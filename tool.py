import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def seleted_site_by_num(data, information, threshold):
    sub = np.size(data,0)
    site_label = np.array(information['site'])
    site = np.unique(site_label)
    seleted_site = []
    seleted_index = []
    index = np.arange(sub)
    for site_i in site:
        temp = index[np.array(information['site'] == site_i)]
        diagnosis = np.array(information['diagnosis'])[information['site'] == site_i]
        tdc = np.sum(diagnosis == 2)
        asd = np.sum(diagnosis == 1)
        sex = np.array(information['sex'])[information['site'] == site_i]
        male_num = np.sum(sex == 2)
        female_num = np.sum(sex == 1)
        age = np.array(information['age'])[information['site'] == site_i]
        max_age = np.max(age)
        min_age = np.min(age)
        mean_age = np.mean(age)
        print("site: ", site_i, ", num: ", np.size(temp, 0)," ASD/TDC: ",asd,"/",tdc, " sex: ",male_num,"/",female_num," age: ",min_age,"/",max_age,"/",mean_age)
        if np.size(temp,0) >= threshold:
            seleted_site.append(site_i)
            seleted_index.extend(list(temp))
    return seleted_site, seleted_index

def seleted_site_by_name(data, information,site_name):
    sub = np.size(data,0)
    site_label = np.array(information['site'])
    site = np.unique(site_label)
    seleted_site = []
    seleted_index = []
    index = np.arange(sub)
    for site_i in site:
        if site_i in site_name:
            temp = index[np.array(information['site'] == site_i)]
            print("site: ", site_i)
            print("number: ", len(temp))
            seleted_site.append(site_i)
            seleted_index.extend(list(temp))
    return seleted_site, seleted_index

def divide_by_sites(feature, information):
    sub = np.size(feature,0)
    index = np.arange(sub)
    site_label = np.array(information['site'])
    site = np.unique(site_label)
    site = np.delete(site, np.where(site == 'CMU'))
    df = pd.DataFrame(index=site, columns=['index_train', 'index_test', 'number'])
    df = df.astype(object)
    for site_i in site:
        temp = np.array(information['site'] == site_i)
        df.loc[site_i, 'index_train'] = index[~temp]
        df.loc[site_i, 'index_test'] = index[temp]
        df.loc[site_i, 'number'] = np.size(index[temp], 0)
    site = []
    train_index = []
    test_index = []
    for site_i in df.index:
        site.append(site_i)
        train_index.append(df.loc[site_i, 'index_train'])
        test_index.append(df.loc[site_i, 'index_test'])

    return zip(site,train_index,test_index)

def evaluate(pred, label):
    pred = np.array(pred, dtype='int')
    label = np.array(label, dtype='int')
    print(pred)
    print(label)
    TP = sum(label[pred == 1])
    FP = len(label[pred == 1]) - sum(label[pred == 1])
    FN = sum(label[pred == 0])
    TN = len(label[pred == 0]) - sum(label[pred == 0])
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1_score = (TP + TP) / (TP + FP + FN + TP)
    return accuracy, sensitivity, precision, F1_score

def get_auc(prob, label):
    AUC = roc_auc_score(label, prob)
    return AUC

def calulate_acc(pred, label):
    acc = (len(label) - np.sum(abs(pred - label))) / len(label)
    return acc


def inter_CV(feature, label, information, k=5, state=1980):
    kf = StratifiedKFold(n_splits=k, random_state=state, shuffle=True)
    sub = np.size(feature,0)
    index = np.arange(sub)
    site_label = np.array(information['site'])
    site = np.unique(site_label)
    train_index = []
    test_index = []
    for i in range(k):
        train_index_i = []
        test_index_i = []
        for site_i in site:
            index_i = index[information['site'] == site_i]
            label_i = label[information['site'] == site_i]
            j = 0
            for train, test in kf.split(index_i, label_i[:, 0]):
                if i == j:
                    train_index_i.extend(index_i[train])
                    test_index_i.extend(index_i[test])
                j = j + 1
        train_index.append(train_index_i)
        test_index.append(test_index_i)
    return zip(train_index, test_index)


def gettri(feature):
    sub = np.size(feature, 0)
    node = np.size(feature, 1)
    index = np.triu_indices(node, k=1, m=None)
    trifeature = np.zeros([sub, int((node * node - node) / 2)])
    for i in range(sub):
        trifeature[i,:] = np.reshape(feature[i,:],[node,node])[index]
    return trifeature
