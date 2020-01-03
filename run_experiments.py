from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn
from arguments import *
import matplotlib.pyplot as plt


import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import keras
from keras import backend as K
tf.set_random_seed(0)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)


from datahelper import DataSet
#import logging
from itertools import product


import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Add
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten, Reshape
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2



TABSY = "\t"
figdir = "figures/"



def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))


    encode_smiles= Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']


    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_onehot.png')

    return interactionModel





def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') 
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel

def deep_categorical_contact_tensor(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') 
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    
    # multiplefeatures = [ {'length': 500, 'filter_length': 32, 'filter_extra_no':3, 'method':eval('Dataset.calculate_energy') }]
    if FLAGS.multiplefeatures:
        multiplefeatures = FLAGS.multiplefeatures
    else:
        multiplefeatures = None
    
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    encode_protein_main = encode_protein
    inputs_multiple= []
    # encode_protein_2 = tf.zeros([encode_protein.shape[0].value, 0])
    # encode_protein_3 = encode_protein_2
    functions_multiple = []
    
    my_iteration = 1 
    if multiplefeatures != None:
        for x in multiplefeatures:
            m,n = combination_feature(FLAGS=FLAGS, 
                FEATURE_LENGTH = x['length'], NUM_FILTERS=NUM_FILTERS, FILTER_LENGTH = x['filter_length'], 
                FILTERS_EXTENSION = x['filter_extra_no']) 
            inputs_multiple.append(n)
            functions_multiple.append(x['method'])
            if my_iteration != 1:
                encode_protein = keras.layers.concatenate([encode_protein, m], axis=-1)
            else:
                encode_protein = m
                my_iteration = 0

        encode_protein_2 = encode_protein
    
    my_iteration = 1

    if FLAGS.tensor != None:
        function_pssm = []
        for x in eval(FLAGS.tensor):
            # m = Input(shape=(x['length'],1), dtype='float32')
            m = Input(shape=(x['length'],) , dtype='float32')
            fx = Reshape((x['length'],1),input_shape=(x['length'],))(m)
            fx = Conv1D(filters=64, kernel_size=3, activation='linear',padding='same')(fx)
            blck = fx
            blck = BatchNormalization()(blck)
            blck = PReLU()(blck)
            encode = GlobalMaxPooling1D()(blck)
            try:
                if x['convnet'] != None:
                    encode, m = combination_feature_1d(
                        FLAGS=FLAGS,
                        FEATURE_LENGTH = x['length'],
                        NUM_FILTERS=x['convnet']['num_filters'],
                        FILTER_LENGTH=x['convnet']['filter_length'],
                        FILTERS_EXTENSION=x['convnet']['filter_extra_no']
                    )
            except:
                pass
            inputs_multiple.extend([m])
            function_pssm.append(x['function'])
            if my_iteration != 1:
                encode_protein = keras.layers.concatenate([encode_protein, encode], axis=-1)
            else:
                encode_protein = encode
                my_iteration = 0
    
        encode_protein_3 = encode_protein
    to_concatenate = ['encode_protein_main', 'encode_protein_2', 'encode_protein_3'][1:]
    encode_protein = eval('encode_protein_main')
    for y in to_concatenate:
        try:
            encode_protein = keras.layers.concatenate([encode_protein , eval(y)] , axis=-1)
        except:
            print("ERROR " + " finding " + y )
            # exit()



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    # FC1 = Dense(1024, activation='relu')(encode_interaction)
    look_back = 1
    encode_interaction =   Reshape(target_shape=(1,encode_interaction.shape.dims[1].value))(encode_interaction)
    FC1 = (LSTM(4, input_shape=(1, look_back)))(encode_interaction)
    # FC2 = Dropout(0.1)(FC1)
    # FC2 = Dense(1024, activation='relu')(FC2)
    # FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC1)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'
    m_inputs = [XDinput, XTinput]
    m_inputs.extend(inputs_multiple)
    interactionModel = Model(inputs=m_inputs, outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical_tensor_contact_new.png')

    if FLAGS.tensor != None:
        functions_multiple = (functions_multiple, function_pssm)

    return interactionModel, functions_multiple



def deep_categorical_contact(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') 
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    
    # multiplefeatures = [ {'length': 500, 'filter_length': 32, 'filter_extra_no':3, 'method':eval('Dataset.calculate_energy') }]
    if FLAGS.multiplefeatures:
        multiplefeatures = FLAGS.multiplefeatures
    else:
        multiplefeatures = None
    
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    inputs_multiple = []
    functions_multiple = []
    if multiplefeatures != None:
        for x in multiplefeatures:
            m,n = combination_feature(FLAGS=FLAGS, 
                FEATURE_LENGTH = x['length'], NUM_FILTERS=NUM_FILTERS, FILTER_LENGTH = x['filter_length'], 
                FILTERS_EXTENSION = x['filter_extra_no']) 
            inputs_multiple.append(n)
            functions_multiple.append(x['method'])
            encode_protein = keras.layers.concatenate([encode_protein, m], axis=-1)

    


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'
    m_inputs = [XDinput, XTinput]
    m_inputs.extend(inputs_multiple)
    interactionModel = Model(inputs=m_inputs, outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical_contact.png')

    return interactionModel, functions_multiple


def combination_feature(FLAGS, FEATURE_LENGTH, NUM_FILTERS, FILTER_LENGTH, FILTERS_EXTENSION):
    XEInput = Input(shape=(FEATURE_LENGTH,1,), dtype='float32')
    energy_mat = Conv1D(filters = NUM_FILTERS, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid')(XEInput)
    for x in range(FILTERS_EXTENSION-1):
        energy_mat = Conv1D(filters=NUM_FILTERS*(x+2), kernel_size=FILTER_LENGTH,  activation='relu', padding='valid')(energy_mat)
    encode_energy = GlobalMaxPooling1D()(energy_mat)

    return encode_energy, XEInput

def combination_feature_1d(FLAGS, FEATURE_LENGTH, NUM_FILTERS, FILTER_LENGTH, FILTERS_EXTENSION):
    XEInput = Input(shape=(FEATURE_LENGTH,1), dtype='float32')
    # XEInput = Reshape(target_shape=(FEATURE_LENGTH,1))(XEInput)
    energy_mat = Conv1D(filters = NUM_FILTERS, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid')(XEInput)
    for x in range(FILTERS_EXTENSION-1):
        energy_mat = Conv1D(filters=NUM_FILTERS*(x+2), kernel_size=FILTER_LENGTH,  activation='relu', padding='valid')(energy_mat)
    encode_energy = GlobalMaxPooling1D()(energy_mat)

    return encode_energy, XEInput


def build_single_drug(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):

    interactionModel = Sequential()
    XTmodel = Sequential()
    XTmodel.add(Activation('linear', input_shape=(FLAGS.target_count,)))


    encode_smiles = Sequential()
    encode_smiles.add(Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SMI_LEN, SMI_EMBEDDING_DIMS)
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(GlobalMaxPooling1D())


    interactionModel.add(Add([encode_smiles, XTmodel], mode='concat', concat_axis=1))
    #interactionModel.add(layers.merge.Concatenate([XDmodel, XTmodel]))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))


    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_drug.png')

    return interactionModel


def build_single_prot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):

    interactionModel = Sequential()
    XDmodel = Sequential()
    XDmodel.add(Activation('linear', input_shape=(FLAGS.drugcount,)))


    XTmodel1 = Sequential()
    XTmodel1.add(Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128,  input_length=FLAGS.max_seq_len))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SEQ_LEN, SEQ_EMBEDDING_DIMS)
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(GlobalMaxPooling1D())


    interactionModel.add(Add([XDmodel, XTmodel1], mode='concat', concat_axis=1))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_protein.png')

    return interactionModel

def build_baseline(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()

    XDmodel = Sequential()
    XDmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.drug_count, )))

    XTmodel = Sequential()
    XTmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.target_count,)))


    interactionModel.add(Add([XDmodel, XTmodel], mode='concat', concat_axis=1))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_baseline.png')

    return interactionModel

def nfold_1_2_3_setting_sample(XD, XT,  Y, label_row_inds, label_col_inds, measure, runmethod,  FLAGS, dataset):

    test_set, outer_train_sets = dataset.read_sets(FLAGS)

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    #logger.info('Start training')
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))



    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds,
                                                                                                measure, runmethod, FLAGS, train_sets, val_sets)

    #print("Test Set len", str(len(test_set)))
    #print("Outer Train Set len", str(len(outer_train_sets)))
    logging("---WITH-TEST_SETS---",FLAGS)
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds,
    measure, runmethod, FLAGS, train_sets, test_sets)

    testperf = all_predictions[bestparamind]##pointer pos

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd 

def deep_lstm(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') 
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    
    # multiplefeatures = [ {'length': 500, 'filter_length': 32, 'filter_extra_no':3, 'method':eval('Dataset.calculate_energy') }]
    if FLAGS.multiplefeatures:
        multiplefeatures = FLAGS.multiplefeatures
    else:
        multiplefeatures = None
    
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    encode_protein_main = encode_protein
    inputs_multiple= []
    # encode_protein_2 = tf.zeros([encode_protein.shape[0].value, 0])
    # encode_protein_3 = encode_protein_2
    functions_multiple = []
    
    my_iteration = 1 
    if multiplefeatures != None:
        for x in multiplefeatures:
            m,n = combination_feature(FLAGS=FLAGS, 
                FEATURE_LENGTH = x['length'], NUM_FILTERS=NUM_FILTERS, FILTER_LENGTH = x['filter_length'], 
                FILTERS_EXTENSION = x['filter_extra_no']) 
            inputs_multiple.append(n)
            functions_multiple.append(x['method'])
            if my_iteration != 1:
                encode_protein = keras.layers.concatenate([encode_protein, m], axis=-1)
            else:
                encode_protein = m
                my_iteration = 0

        encode_protein_2 = encode_protein
    
    my_iteration = 1

    if FLAGS.tensor != None:
        function_pssm = []
        for x in eval(FLAGS.tensor):
            # m = Input(shape=(x['length'],1), dtype='float32')
            m = Input(shape=(x['length'],) , dtype='float32')
            fx = Reshape((x['length'],1),input_shape=(x['length'],))(m)
            fx = Conv1D(filters=64, kernel_size=3, activation='linear',padding='same')(fx)
            blck = fx
            blck = BatchNormalization()(blck)
            blck = PReLU()(blck)
            encode = GlobalMaxPooling1D()(blck)
            try:
                if x['convnet'] != None:
                    encode, m = combination_feature_1d(
                        FLAGS=FLAGS,
                        FEATURE_LENGTH = x['length'],
                        NUM_FILTERS=x['convnet']['num_filters'],
                        FILTER_LENGTH=x['convnet']['filter_length'],
                        FILTERS_EXTENSION=x['convnet']['filter_extra_no']
                    )
            except:
                pass
            inputs_multiple.extend([m])
            function_pssm.append(x['function'])
            if my_iteration != 1:
                encode_protein = keras.layers.concatenate([encode_protein, encode], axis=-1)
            else:
                encode_protein = encode
                my_iteration = 0
    
        encode_protein_3 = encode_protein
    to_concatenate = ['encode_protein_main', 'encode_protein_2', 'encode_protein_3'][1:]
    encode_protein = eval('encode_protein_main')
    for y in to_concatenate:
        try:
            encode_protein = keras.layers.concatenate([encode_protein , eval(y)] , axis=-1)
        except:
            print("ERROR " + " finding " + y )
            # exit()



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    # FC1 = Dense(1024, activation='relu')(encode_interaction)
    look_back = 1
    encode_interaction =   Reshape(target_shape=(1,encode_interaction.shape.dims[1].value))(encode_interaction)
    FC1 = (LSTM(4, input_shape=(1, look_back)))(encode_interaction)
    # FC2 = Dropout(0.1)(FC1)
    # FC2 = Dense(1024, activation='relu')(FC2)
    # FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC1)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'
    m_inputs = [XDinput, XTinput]
    m_inputs.extend(inputs_multiple)
    interactionModel = Model(inputs=m_inputs, outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/lstm.png')

    if FLAGS.tensor != None:
        functions_multiple = (functions_multiple, function_pssm)

    return interactionModel, functions_multiple




def make_features_set(data,functions):
    feat = []
    for x in functions:
        feat_y = []
        values={}
        for z in np.unique(data):
            values[z] = x(z)
        for y in data:
            feat_y.append(values[y])
        feat.append(feat_y)
    return feat

def make_pssm_features_set(data, functions, dataset=None):
    feat = []
    for x in functions:
        feat_y = []
        values={}
        x = getattr(dataset,x)
        for z in np.unique(data):
            values[z] = x(z)
        for y in data:
            feat_y.append(values[y])
        feat.append(feat_y)
    return feat

def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): 

    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                 #100
    batchsz = FLAGS.batch_size                             #256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        #print("trrows", str(trrows), str(len(trrows)))
        #print("trcols", str(trcols), str(len(trcols)))

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)


        pointer = 0

        for param1ind in range(len(paramset1)): #hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)): #learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    # if gridmodel is tuple, we have to convert protein sets to the respective feature sets as well
                    if type(gridmodel) != eval("tuple"):
                        gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch,
                                validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False )
                        gridmodel.save_weights("model.h5")
                        predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                        loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                    else:
                        if type(gridmodel[1]) != tuple:
                            # make_features_set([np.array(train_drugs), np.array(train_prots)], gridmodel[1])
                            trainset=[np.array(train_drugs), np.array(train_prots)]
                            for r in make_features_set([list(FLAGS.protein.keys())[r] for r in trcols], gridmodel[1]):
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                trainset.append(r)
                            valset = [np.array(val_drugs), np.array(val_prots) ]
                            for r in make_features_set([list(FLAGS.protein.keys())[r] for r in tecols], gridmodel[1]) :
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                valset.append(r) 
                            # np.concatenate(( [train_drugs, train_prots], make_features_set([list(FLAGS.protein.keys())[r] for r in trcols], gridmodel[1]) ))
                            gridres = gridmodel[0].fit( trainset
                                , np.array(train_Y), batch_size=batchsz, epochs=epoch,
                                # np.concatenate(( [np.array(val_drugs), np.array(val_prots) ], make_features_set([list(FLAGS.protein.keys())[r] for r in tecols], gridmodel[1]) ))
                                validation_data = (valset, np.array(val_Y))
                            )
                            gridmodel[0].save_weights("model.h5")
                            predicted_labels = gridmodel[0].predict(valset)
                            loss, rperf2 = gridmodel[0].evaluate(valset, np.array(val_Y), verbose=0)
                        else:
                            trainset=[np.array(train_drugs), np.array(train_prots)]
                            # processing for sequence related set
                            for r in make_features_set([list(FLAGS.protein.keys())[r] for r in trcols], gridmodel[1][0]):
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                trainset.append(r)
                            #processing for pssm related set
                            for r in make_pssm_features_set([list(FLAGS.protein.keys())[r] for r in trcols], gridmodel[1][1], dataset = FLAGS.data_instance):
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                trainset.append(r)

                            valset = [np.array(val_drugs), np.array(val_prots) ]
                            for r in make_features_set([list(FLAGS.protein.keys())[r] for r in tecols], gridmodel[1][0]) :
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                valset.append(r) 
                            for r in make_pssm_features_set([list(FLAGS.protein.keys())[r] for r in tecols], gridmodel[1][1], dataset = FLAGS.data_instance):
                                r = np.array(r)
                                r = r.reshape(-1,r.shape[1],1)
                                valset.append(r)
                            
                            # np.concatenate(( [train_drugs, train_prots], make_features_set([list(FLAGS.protein.keys())[r] for r in trcols], gridmodel[1]) ))

                            gridres = gridmodel[0].fit( trainset
                                , np.array(train_Y), batch_size=batchsz, epochs=epoch,
                                # np.concatenate(( [np.array(val_drugs), np.array(val_prots) ], make_features_set([list(FLAGS.protein.keys())[r] for r in tecols], gridmodel[1]) ))
                                validation_data = (valset, np.array(val_Y))
                            )
                            gridmodel[0].save_weights("model.h5")
                            predicted_labels = gridmodel[0].predict(valset)
                            loss, rperf2 = gridmodel[0].evaluate(valset, np.array(val_Y), verbose=0)
                            
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]


                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" %
                    (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss), FLAGS)

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss

                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):

                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1

    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses



def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select



def plotLoss(history, batchind, epochind, param3ind, foldind):

    figname = "b"+str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_"  + str( foldind) + "_" + str(time.time())
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)



def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity



def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.dataset_path, 
                      setting_no = FLAGS.problem_type, 
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size
    
    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(XD.shape)
    targetcount = XT.shape[0]
    print(XT.shape)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    FLAGS.multiplefeatures = eval(FLAGS.multiplefeatures)
    multiplefeatures = (FLAGS.multiplefeatures)
    FLAGS.protein = dataset.proteins
    FLAGS.data_instance = dataset
    
    if multiplefeatures != None:
        dataset.get_proteins_features()
        for x in range(len(multiplefeatures)):
            try:
                func = getattr(dataset, multiplefeatures[x]['method'])  # method will not work: Use 'function' instead
                multiplefeatures[x]['method'] = func
            except AttributeError:
                print("Bro... You mentioned wrong dataset function name in method")
                exit()
    dataset.set_pssm_dir('/anup_files/FILES/programs/drug/DeepDTA/pssm-repo/pssm_data/output')

    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
        perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
        (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


 

def run_regression( FLAGS ):

    perfmeasure = get_cindex
    deepmethod = eval(FLAGS.method)

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

logging(str(FLAGS), FLAGS)
run_regression(FLAGS)
