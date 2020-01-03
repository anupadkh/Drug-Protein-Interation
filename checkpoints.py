import keras
from keras.callbacks import CSVLogger

def checkpoint_with_fname(filepath):
    return [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
        keras.callbacks.CSVLogger(filepath+'.csv', append=True),

    ]
