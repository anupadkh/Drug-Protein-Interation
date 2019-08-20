"""
This code is for the final project of biomedical course
Implementation of paper "protein classification using modified NGSG model"
Input file is 'fasta' format file
Student: Zhen Liu
NID: zh116855
"""
import pandas as pd
import numpy as np
# from numpy import *
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import svm, metrics

def split_data(feature, size):
    """
    This function is for integrating all features into a big list
    Transform feature list into array
    Split all feature from dataset into training set and test set in the ratio of 7:3
    """
    X = []
    Y = []
    for i in range(len(feature)):
        X.append(feature[i][1])
        Y.append(feature[i][0])
    X = np.array(X)
    Y = np.array(Y).flatten()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=1)
    return x_train, x_test, y_train, y_test

def denoise(feature, n_pos):
    """
    This function is for denoising all features whose postion are detected by function noise_pos
    """
    # Slicing the data the indexes in n_pos
    new_feature = []
    for i in range(len(feature)):
        f_vec = []
        for j in range(1900):               # Maximum of n_pos = 1899
            if j not in n_pos:
                f_vec.append(feature[i][1][j])
        new_feature.append((feature[i][0], f_vec))
    return new_feature

def noise_pos(feature, thres):
    """
    This function is for creating position list
    The list contains those features occurs less than given number
    """
    n_pos = []
    for i in range(len(feature[0][1])):
        max_num = 0
        count = 0
        for j in range(len(feature)):
            max_num = max(feature[j][1][i], max_num)
            count += feature[j][1][i]
        if max_num < thres:
            n_pos.append(i)
    return n_pos

def feat_extract(aa, seq, buffer):
    """
    This function is for integrating features and tags into a list
    """
    full_feature = {}
    feature = []
    for key in seq:
        bi_gram = vec_init(seq, aa)
        full_feature[key] = feat_count(key, seq, bi_gram, buffer)
    for key in full_feature:
        feature.append((tag_extract(key), list(full_feature[key].values())))
    # print(feature)
    return feature

def feat_count(key, seq, table, buffer):
    """
    This function is for counting feature numbers including k-skip-bi-gram and relative position
    """
    acid_list = list(seq[key])
    for i in range(len(acid_list)-(buffer+1)):
        grams = skip_buf(buffer, i, acid_list)
        for gram in grams:
            table[gram] += 1
        pos = pos_cal(seq[key], i, buffer)
        temp = acid_list[i] + str(pos)
        table[temp] += 1
    return table

def pos_cal(seq, curr_pos, buffer):
    """
    Calculating relative position to C-terminus by given current position and buffer number
    """
    dist = len(seq) - curr_pos
    pos = dist + (buffer - dist)%buffer
    return pos

def skip_buf(buffer, pos, acid_list):
    """
    Adding buffer into skip-gram model
    """
    grams = []
    for i in range(1, buffer+2):
        grams.append(acid_list[pos] + acid_list[pos+i])
    return grams

def buf_num(skip_num, para_a):
    """
    This function is for calculating buffer numbers by given parameter
    """
    c = skip_num + (para_a - skip_num) % para_a
    return c

def tag_extract(str):
    """
    This function is for extracting tags from every sequence procedings
    """
    tag = ''
    for char in list(str):
        if char != '|':
            tag += char
        else:
            break
    return tag

def vec_init(seq, amino_acid):
    """
    This function is for initialize feature stats table
    """
    bi_gram = {}
    for acid_1 in amino_acid:
        for acid_2 in amino_acid:
            temp_1 = acid_1 + acid_2
            bi_gram[temp_1] = 0
    pos_list = pos_init(seq, amino_acid)
    for pos_motif in pos_list:
        bi_gram[pos_motif] = 0
    return bi_gram

def pos_init(seq, aa):
    """
    This function is for initializing realative position list
    """
    pos_list = []
    for a in aa:
        for i in range(count_len(seq) + 10):
            temp = a + str(i)
            pos_list.append(temp)
    return pos_list

def count_len(seq):
    """
    Saving the max length of sequence in the dataset
    """
    length = 0
    for key in seq:
        length = max(length, len(seq[key]))
    return length

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

def make_feature_set(protseq, buffer=1):
    aa = list(CHARPROTSET.keys())
    bi_gram = vec_init(protseq, aa)
    un_feature = feat_extract(aa,protseq,buffer)
    no_pos = noise_pos(un_feature, 2)
    feature = denoise(un_feature, no_pos)
    remaining_bi_grams = np.delete(list(bi_gram.keys()), no_pos)
    return remaining_bi_grams, feature, bi_gram.keys(), un_feature

def from_feature_to_DataFrame(feature):
    y = []
    for x in feature:
        y.append(x[1])
    return pd.DataFrame(y)

def removed_unrequired_aminos(bigram, feature=None, buffer=1, aa = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P'], prot_aa = CHARPROTSET.keys()):

    l = [z for z in prot_aa if z not in aa]
    # l = ['B', 'O', 'U', 'X', 'Z']   # They don't belong to our 20 amino acids
    # b,f,b_org, f_org = make_feature_set(protseq, buffer)
    listed = list(bigram)
    indices = [listed.index(i) for i in listed if (i[0] in l or i[1] in l)]
    if feature:
        feature = [x[1] for x in feature]
        return np.delete(bigram,indices), np.delete(feature,indices, axis=1)
    else:
        return np.delete(bigram,indices)

def get_contact_mappings(bigram_array, matrix):
    a = dict()
    for i in removed_unrequired_aminos(bigram_array):
        a[i] = matrix.loc[i[0], i[1]]
    return a, [a[i] for i in removed_unrequired_aminos(bigram_array)]

def get_energy_matrix(feature, contacts):
    x = []
    for feat in feature:
        y = sorted(list(zip(contacts.keys(), feat)), key = lambda x:x[1])
        m = [0 if (r[1] == 0) else contacts[r[0]] for r in y]
        m.reverse()
        x.append(m)
    return x
