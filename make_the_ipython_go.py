import time
import run_experiments as ret
FLAGS=ret.argparser() 
from argparse import Namespace
fpath = "data/kiba/"
setting_no = 1
seqlen = 1000
smilen = 100
dataset = ret.DataSet(fpath=fpath, setting_no=setting_no, seqlen=seqlen, smilen=smilen)
FLAGS = eval("Namespace(batch_size=256, binary_th=0.0, feat_length=400, checkpoint_path='', dataset_path=fpath, is_log=0, learning_rate=0.001, log_dir='logs/', max_seq_len=400, max_smi_len=100, num_classes=0, num_epoch=100, num_hidden=0, num_windows=32, problem_type=1, residues_vectors='..', seq_window_lengths=[8, 12], smi_window_lengths=[4, 8])")
FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
FLAGS.charseqset_size = dataset.charseqset_size
FLAGS.charsmiset_size = dataset.charsmiset_size
NUM_FILTERS = FLAGS.num_windows     # paramset1
# FILTER_LENGTH1 = FLAGS.
FILTER_LENGTH1 = FLAGS.smi_window_lengths      #paramset2                         #[4, 8]#[4,  32] #[4,  8] #filter length smi
FILTER_LENGTH2 = FLAGS.seq_window_lengths    #paramset3
multiplefeatures = [
    # Energy Set
    {
        'length': 400,
        'filter_length': 16,
        'filter_extra_no': 3,
    }
]
# for x in multiplefeatures: 
#     y = ret.combination_feature(FLAGS=FLAGS,  
#         FEATURE_LENGTH = x['length'], NUM_FILTERS=NUM_FILTERS, FILTER_LENGTH = x['filter_length'],  
#         FILTERS_EXTENSION = x['filter_extra_no']
#     )  
                        