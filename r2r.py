import pandas as pd
import numpy as np
import itertools

R2R = [
  'I', 'V', 'L', 'F', 'C', 'M', 'A', 'G', 'T', 'S', 'W', 'Y', 'P', 'H', 'E', 'Q', 'D', 'N', 'K', 'R'
]

def pair(s1, s2):
    a = []
    for x in s1:
        for y in s2:
            a.append((x,y))

    return a

class RSRV(object):
    pc = []
    file_path = "../"

    def initialize(self, file_path = "../", file_name = "PC_RSRV"):
        self.file_path = file_path
        for x in range(1,7):
            self.pc.append(pd.read_csv(file_path + file_name + str(x) + '.csv' , index_col = 0)) 
    
    def parse(self, x , y):
        z = []
        for pc_idx in self.pc:
            z.append( (pc_idx.loc[x,y] + pc_idx.loc[y,x])/2 )

        return z

    def RSVR_feature( self, residue_seq1, residue_seq2):
        # f = list(set(itertools.combinations(residue_seq1+residue_seq2, 2)))
        fv = []
        unique_residues = list(pair(residue_seq1 , residue_seq2))
        for x,y in unique_residues:
            fv += self.parse(x,y)
        return fv


x = RSRV()
x.initialize(file_path="./PC/")
print(x.RSVR_feature('PVKAAFV', 'EFAVLTI'))