import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
#from keras.preprocessing.sequence import pad_sequences


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

energy = {'A': [-1.65, -2.83, 1.16, 1.8, -3.73, -0.41, 1.9, -3.69, 0.49, -3.01, -2.08, 0.66, 1.54, 1.2, 0.98, -0.08, 0.46, -2.31, 0.32, -4.62], 'C': [-2.83, -39.58, -0.82, -0.53, -3.07, -2.96, -4.98, 0.34, -1.38, -2.15, 1.43, -4.18, -2.13, -2.91, -0.41, -2.33, -1.84, -0.16, 4.26, -4.46], 'D': [1.16, -0.82, 0.84, 1.97, -0.92, 0.88, -1.07, 0.68, -1.93, 0.23, 0.61, 0.32, 3.31, 2.67, -2.02, 0.91, -0.65, 0.94, -0.71, 0.9], 'E': [1.8, -0.53, 1.97, 1.45, 0.94, 1.31, 0.61, 1.3, -2.51, 1.14, 2.53, 0.2, 1.44, 0.1, -3.13, 0.81, 1.54, 0.12, -1.07, 1.29], 'F': [-3.73, -3.07, -0.92, 0.94, -11.25, 0.35, -3.57, -5.88, -0.82, -8.59, -5.34, 0.73, 0.32, 0.77, -0.4, -2.22, 0.11, -7.05, -7.09, -8.8], 'G': [-0.41, -2.96, 0.88, 1.31, 0.35, -0.2, 1.09, -0.65, -0.16, -0.55, -0.52, -0.32, 2.25, 1.11, 0.84, 0.71, 0.59, -0.38, 1.69, -1.9], 'H': [1.9, -4.98, -1.07, 0.61, -3.57, 1.09, 1.97, -0.71, 2.89, -0.86, -0.75, 1.84, 0.35, 2.64, 2.05, 0.82, -0.01, 0.27, -7.58, -3.2], 'I': [-3.69, 0.34, 0.68, 1.3, -5.88, -0.65, -0.71, -6.74, -0.01, -9.01, -3.62, -0.07, 0.12, -0.18, 0.19, -0.15, 0.63, -6.54, -3.78, -5.26], 'K': [0.49, -1.38, -1.93, -2.51, -0.82, -0.16, 2.89, -0.01, 1.24, 0.49, 1.61, 1.12, 0.51, 0.43, 2.34, 0.19, -1.11, 0.19, 0.02, -1.19], 'L': [-3.01, -2.15, 0.23, 1.14, -8.59, -0.55, -0.86, -9.01, 0.49, -6.37, -2.88, 0.97, 1.81, -0.58, -0.6, -0.41, 0.72, -5.43, -8.31, -4.9], 'M': [-2.08, 1.43, 0.61, 2.53, -5.34, -0.52, -0.75, -3.62, 1.61, -2.88, -6.49, 0.21, 0.75, 1.9, 2.09, 1.39, 0.63, -2.59, -6.88, -9.73], 'N': [0.66, -4.18, 0.32, 0.2, 0.73, -0.32, 1.84, -0.07, 1.12, 0.97, 0.21, 0.61, 1.15, 1.28, 1.08, 0.29, 0.46, 0.93, -0.74, 0.93], 'P': [1.54, -2.13, 3.31, 1.44, 0.32, 2.25, 0.35, 0.12, 0.51, 1.81, 0.75, 1.15, -0.42, 2.97, 1.06, 1.12, 1.65, 0.38, -2.06, -2.09], 'Q': [1.2, -2.91, 2.67, 0.1, 0.77, 1.11, 2.64, -0.18, 0.43, -0.58, 1.9, 1.28, 2.97, -1.54, 0.91, 0.85, -0.07, -1.91, -0.76, 0.01], 'R': [0.98, -0.41, -2.02, -3.13, -0.4, 0.84, 2.05, 0.19, 2.34, -0.6, 2.09, 1.08, 1.06, 0.91, 0.21, 0.95, 0.98, 0.08, -5.89, 0.36], 'S': [-0.08, -2.33, 0.91, 0.81, -2.22, 0.71, 0.82, -0.15, 0.19, -0.41, 1.39, 0.29, 1.12, 0.85, 0.95, -0.48, -0.06, 0.13, -3.03, -0.82], 'T': [0.46, -1.84, -0.65, 1.54, 0.11, 0.59, -0.01, 0.63, -1.11, 0.72, 0.63, 0.46, 1.65, -0.07, 0.98, -0.06, -0.96, 1.14, -0.65, -0.37], 'V': [-2.31, -0.16, 0.94, 0.12, -7.05, -0.38, 0.27, -6.54, 0.19, -5.43, -2.59, 0.93, 0.38, -1.91, 0.08, 0.13, 1.14, -4.82, -2.13, -3.59], 'W': [0.32, 4.26, -0.71, -1.07, -7.09, 1.69, -7.58, -3.78, 0.02, -8.31, -6.88, -0.74, -2.06, -0.76, -5.89, -3.03, -0.65, -2.13, -1.73, -12.39], 'Y': [-4.62, -4.46, 0.9, 1.29, -8.8, -1.9, -3.2, -5.26, -1.19, -4.9, -9.73, 0.93, -2.09, 0.01, 0.36, -0.82, -0.37, -3.59, -12.39, -2.68]}


CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1 

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def turn_energy_set(line, MAX_SEQ_LEN, smi_ch_ind):
  X = np.zeros((MAX_SEQ_LEN, 21)) 
  for i, ch in enumerate(line[:MAX_SEQ_LEN]):
    X[i] = [smi_ch_ind[ch]]  + energy[ch]

  return X

## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle = False):
    self.SEQLEN = seqlen #1000
    self.SMILEN = smilen # 100
    #self.NCLASSES = n_classes
    self.charseqset = CHARPROTSET
    self.charseqset_size = CHARPROTLEN

    self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
    self.charsmiset_size = CHARISOSMILEN
    self.PROBLEMSET = setting_no

    # read raw file
    # self._raw = self.read_sets( FLAGS)

    # iteration flags
    # self._num_data = len(self._raw)


  def read_sets(self, FLAGS): ### fpath should be the dataset folder /kiba/ or /davis/
    fpath = FLAGS.dataset_path
    setting_no = FLAGS.problem_type
    print("Reading %s start" % fpath)

    test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))
    
    return test_fold, train_folds

  def parse_data(self, FLAGS,  with_label=True, with_energy=False): 
    fpath = FLAGS.dataset_path	
    print("Read %s start" % fpath)

    ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
    if FLAGS.is_log:
        Y = -(np.log10(Y/(math.pow(10,9))))

    XD = []
    XT = []
    if (with_label and with_energy):
      for d in ligands.keys():
        XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))
        
      for t in proteins.keys():
        XT.append(turn_energy_set(proteins[t], self.SEQLEN, self.charseqset))
    elif with_label:
        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
    else:
        for d in ligands.keys():
            XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))
  
    return XD, XT, Y




