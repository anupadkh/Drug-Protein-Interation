import itertools
import random

from itertools import combinations
import numpy as np
import math
#LG is some predefined constant
LG =  5
import math

def trigrams(matrix):
    return_matrix = []
    for x in range(len(matrix[0])):
      for y in range(len(matrix[0])):
        for z in range(len(matrix[0])):
          for i in range( len(matrix) - 2):
            value = matrix[i][x] * matrix[i+1][y] * matrix[i+2][z]
            return_matrix.append( value )
    return return_matrix

def residue_probing_transform(matrix):
  features = []
  #lets assume matrix[0] is 20 which is generally the case
  #sum of nth aa group in pssm
  matrix_sums = list(map(sum, zip(*matrix)))
  res_trnfr = []
  for i in range(len(matrix[0])):
    for j in range(len(matrix[0])):
      features.append( ( matrix_sums[i]+ matrix_sums[j]) / len(matrix) )
  return features

def evolutionary_diff(matrix, d = 30):
  features = []
  #assume all features ares same len and = 20
  for x in range(len(matrix[0] ) ):
    for y in range( len(matrix[0] ) ):
      value = 0
      for i in range( len(matrix) - d ):
        value += (matrix[i][x] - matrix[i+d][y]) ** 2
      value = value / (len(matrix) - d)
      features.append(value )
  #print(len(features) )
  return features
def pssm_sdt_func(matrix):
    #construct empty matrix of size 20 * LG
    pssm_sdt = [[0 for x in range( 20 )] for y in range(LG) ]
    #loop through all i's i.e amino acids
    for lg in range(0, LG):
        for i in range(len(matrix[0] ) ): #number of features
            pssm_sdt[lg][i] = 0
            for j in range(0, len(matrix) - lg):
                pssm_sdt[lg][i] += matrix[j][i] * matrix[j + lg][i] / (len(matrix) - lg)
    return pssm_sdt


#pssm distance transformation of different proteins
def pssm_ddt_func(matrix):
    #lets first calculate a permutations of all indexes in the matrix
    #we can calculate indices for pssm_ddt later on
    a = [x for x in range(20 )] # 20 is total number of amino acids
    #generates matrix of size 380
    indices_matrix = list(itertools.permutations(a, 2)) #combinations of two residues

    #empty matrix
    pssm_ddt = [[0 for x in range(19 * 20 )] for y in range(LG)]

    for lg in range(0, LG):
        for i1 in range(len(matrix[1])): #number of features
            for i2 in range(len(matrix[1])) :
                if i1 == i2:
                    continue
                index = indices_matrix.index((i1, i2))
                for j in range(0, len(matrix) - lg):
                    pssm_ddt[lg][index] += matrix[j][i1] * matrix[j+ lg][i2] / (len(matrix) - lg )
    return pssm_ddt

def feature_space(matrix, pssm_sdt=True, pssm_ddt=True):
    pssm_ddt1 = []
    pssm_sdt1 = []
    if pssm_ddt:
      pssm_ddt1 = [item for sublist in  pssm_ddt_func(matrix) for item in sublist]
    if pssm_sdt:
      pssm_sdt1 = [item for sublist in pssm_sdt_func(matrix) for item in sublist]
    
    ## add the length of matrix( protein ) as a feature
    return pssm_sdt1 + pssm_ddt1 #+ [len(matrix) ]# + trigram_1  + res_trnsfm#+ pssm_sdt1 + pssm_ddt1 #+ trigram_1 # + [random.random() for x in range(300)]

import sys
def read_pssm(pssm_file):
	# this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.

	# index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
	idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)

	# open the two files, read in their data and then close them
	if pssm_file == 'STDIN': fp = sys.stdin
	else: fp = open(pssm_file, 'r')
	lines = fp.readlines()
	fp.close()

	# declare the empty dictionary with each of the entries
	aa = []
	pssm = []

	# iterate over the pssm file and get the needed information out
	for line in lines:
		split_line = line.split()
		# valid lines should have 32 points of data.
		# any line starting with a # is ignored
		try: int(split_line[0])
		except: continue

		if line[0] == '#': continue

		aa_temp = split_line[1]
		aa.append(aa_temp)
		if len(split_line) in (44,22):
			pssm_temp = [-float(i) for i in split_line[2:22]]
		elif len(line) > 70:  # in case double digits of pssm
			pssm_temp = [-float(line[k*3+9: k*3+12]) for k in range(20)]
			pass
		else: continue
		pssm.append([pssm_temp[k] for k in idx_res])

	return aa, pssm