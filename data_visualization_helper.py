import pandas as pd

src_data = '/anup_files/FILES/programs/drug/DeepDTA/source/data/kiba/'

files = {0: 'ligands_iso.txt',
 1: 'proteins.txt',
#  2: '.DS_Store',
#  3: 'folds',
 4: 'kiba_binding_affinity_v2.txt',
 5: 'ligands_can.txt',
 6: 'kiba_drug_sim.txt',
 7: 'kiba_target_sim.txt',
 8: 'Y'
}

# filed = open(src_data + files[1], 'r')
# data = filed.read()

# prot = eval(data)

# filed.close()
# filed = open( src_data + files[5], 'r')
# data = filed.read()

# ligands = eval(data)
# filed.close()

# filed = open( src_data + files[0], 'r')
# data = filed.read()

# ligands_isomers = eval(data)
# filed.close()
# /anup_files/FILES/programs/drug/DeepDTA/CHEM.csv
# library_chem = pd.read_csv(src_data + '../../../' + '/CHEM.csv',header=[0],index_col=0,sep=";") 



# binding_affinity = pd.read_csv( src_data + files[4], header=None, sep="\t")
# del binding_affinity[binding_affinity.columns[-1]]

similarity_drug_ = pd.read_csv( src_data + files[6], header=None, sep="\t")
del similarity_drug_[similarity_drug_.columns[-1]]

# similarity_target_ = pd.read_csv( src_data + files[7], header=None, sep="\t")

import matplotlib.pyplot as plt
pd.plotting.scatter_matrix(similarity_drug_,alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.show()