import os
import datetime
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from keras.utils import multi_gpu_model
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Input, Reshape, Dropout, Flatten, Lambda
from keras import backend
from keras.optimizers import Adam
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization, concatenate, RepeatVector
import tensorflow as tf


# user modules
from trackml.score  import score_event
from trackml.dataset import load_event, load_dataset 
import merge as merge
import time

TRAIN_DATA = '../input/train_1'
TRAIN_NPY = '../input/train_npy'
GPU = 4
BATCH_SIZE = 2048
EPOCHS = 50
INIT_LR = 0.001
DECAY = 0.0
CLASS = 21 
NUM_POINTS = 100000
TRAIN_EVENT = 1
VAL_EVENT = 1
TEST_EVENT = 1

def test_gpu():
    found_gpus = backend.tensorflow_backend._get_available_gpus()
    print("Found GPUs: " + str(found_gpus))


def load_one_event_data(event_id, path=TRAIN_DATA):
    particles = pd.read_csv(os.path.join(path, 'event%s-particles.csv'%event_id))
    hits  = pd.read_csv(os.path.join(path, 'event%s-hits.csv' %event_id))
    truth = pd.read_csv(os.path.join(path, 'event%s-truth.csv'%event_id))
    cells = pd.read_csv(os.path.join(path, 'event%s-cells.csv'%event_id))
    truth = truth.merge(hits,       on=['hit_id'],      how='left')
    truth = truth.merge(particles,  on=['particle_id'], how='left')

    return (truth)


def generate_df(df):
    df = df.copy()
    df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(a   = np.arctan2(df.y, df.x))
    df = df.assign(cosa= np.cos(df.a))
    df = df.assign(sina= np.sin(df.a))
    df = df.assign(phi = np.arctan2(df.z, df.r))
    df = df.assign(z1 = df.z/df.r)
    df = df.assign(z2 = df.z/df.d)
    df = df.assign(z3 = np.log1p(np.absolute(df.z/df.r))*np.sign(df.z))
    df = df.assign(xr = df.x/df.r)
    df = df.assign(yr = df.y/df.r)
    df = df.assign(xd = df.x/df.d)
    df = df.assign(yd = df.y/df.d)
    


    return df

def generate_train_labels(df):
    labels = [0] * len(df.index)
    df['abs_z'] = df.z.abs()
    abs_z = df.abs_z.values
    p = df['particle_id'].values.astype(np.int64)
    particle_ids = list(df.particle_id.unique())
    
    for particle_id in particle_ids:
        if particle_id==0: continue
        particle_indices = np.where(p==particle_id)[0]
        if len(particle_indices) < 4: continue
        particle_indices = particle_indices[np.argsort(abs_z[particle_indices])]

        track_hit_class = 1
        for index in particle_indices:
            labels[index] = track_hit_class
            track_hit_class = track_hit_class + 1
            if track_hit_class > CLASS: break

    df['ytrain'] = labels
    return df


def generate_train_batch(df):
    df = generate_df(df)
    df = generate_train_labels(df)
    x, y, z, a, r, d, sina, cosa, phi, z1, z2, z3, xr, yr, xd, yd, hit_id = \
    df[['x', 'y', 'z', 'a', 'r', 'd', 'sina', 'cosa', 'phi', 'z1', 'z2', 'z3', 'xr', 'yr', 'xd', 'yd', 'hit_id']].values.astype(np.float32).T
    
    input  = np.column_stack((x/1000, y/1000, z/3000, a, r/1000, d/3000, sina, cosa, phi, z1, z2, z3, xr, yr, xd, yd))

    x_train = input[:NUM_POINTS, :]
    y_train = np.expand_dims(df.ytrain.values[:NUM_POINTS], axis=-1)

    if x_train.shape[0] < NUM_POINTS:
        x_train = np.pad(x_train, ( (0, NUM_POINTS-x_train.shape[0]), (0,0)), 'constant')
        y_train = np.pad(y_train, ( (0, NUM_POINTS-y_train.shape[0]), (0,0)), 'constant')
    x_train = np.expand_dims(x_train, axis=0)
    y_train = np.expand_dims(y_train, axis=0)
    

    return x_train, y_train


# def mix_tracks(tracks, noise_hits, input_length):
#     for i, track in enumerate(tracks):
#         if track.shape[0] < input_length:
#             if i+1 < len(tracks)-1:
#                 next_track = tracks[i+1]
#             else:
#                 next_track = track[0]
#             track, label = mix_track_with_noise(track, next_track, noise_hits, input_length)
#             if i == 0:
#                 x_train = np.expand_dims(track, axis=-1)
#                 y_train = np.expand_dims(label, axis=-1)
                           
#             else:
#                 x_train = np.dstack((x_train, track))
#                 y_train = np.dstack((y_train, label))
    
#     x_train = np.transpose(x_train, (2, 0, 1))
#     y_train = np.transpose(y_train, (2, 0, 1))
#     return x_train, y_train

# def mix_track_with_noise(track, next_track, noise_hits, input_length): 
#     idx = np.random.randint(noise_hits.shape[0], size=input_length)
#     #track_positions = random.sample(range(0,input_length), track.shape[0])
#     #track_positions = np.sort(track_positions)
   
#     # 24 hits x 5 features
#     noisy_track = np.zeros((input_length, track.shape[1]))
#     label = np.zeros((input_length, 1))
#     track_idx = 0
#     noise_idx = 0
#     next_track_idx = 0
#     for i in range(input_length):
#         if i and track_idx < track.shape[0]:
#             noisy_track[i] = track[track_idx]
#             track_idx = track_idx+1
#             label[i] = 1
#         else:
#             #adding the hits from another track but it should not loose the majority
#             if next_track_idx < next_track.shape[0] and next_track_idx < track.shape[0]-1:
#                 noisy_track[i] = next_track[next_track_idx]   
#                 next_track_idx = next_track_idx + 1 
#             else:
#                 noisy_track[i] = noise_hits[idx[noise_idx],:]
#                 noise_idx = noise_idx+1
#             label[i] = 0
#     print(label)
#     return noisy_track, label
            


def generate_multiple_event_data(skip=0, nevents=10):
    start = 1000
    x_train_batch = None
    y_train_batch = None
    for i in range(nevents):
        try:
            df = load_one_event_data('00000' + "{:04}".format(start+skip+i))
            print('Generating or loading x_train, y_train of the event00000' + "{:04}".format(start+skip+i))
            x_train_file = os.path.join(TRAIN_NPY, 'event'+'00000' + "{:04}".format(start+skip+i)+'_x_train.npy')
            y_train_file = os.path.join(TRAIN_NPY, 'event'+'00000' + "{:04}".format(start+skip+i)+'_y_train.npy')
            
            if os.path.exists(x_train_file):
                x_train = np.load(x_train_file)
                y_train = np.load(y_train_file)
            else:
                x_train, y_train = generate_train_batch(df)
                np.save(x_train_file, x_train)
                np.save(y_train_file, y_train)
            if i==0:
                x_train_batch = x_train
                y_train_batch = y_train
            else:
                x_train_batch = np.vstack((x_train_batch, x_train))
                y_train_batch = np.vstack((y_train_batch, y_train))
        except:
            pass


    print(x_train_batch.shape)
    print(y_train_batch.shape)
    return x_train_batch, y_train_batch


def draw_train_history(history, draw_val=True, figsize=(12,5)):
    """Make plots of training and validation losses and accuracies"""
    fig = plt.figure(figsize=figsize)
    # Plot loss
    plt.subplot(121)
    plt.plot(history.epoch, history.history['loss'], label='Training set')
    if draw_val:
        plt.plot(history.epoch, history.history['val_loss'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.epoch, history.history['acc'], label='Training set')
    if draw_val:
        plt.plot(history.epoch, history.history['val_acc'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.title('Training accuracy')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show()
    return fig


def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

#https://github.com/garyloveavocado/pointnet-keras/blob/master/train_cls.py
#https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls.py
#FIXME for classification, I haven't fixed the mismatched dimension bug
def build_model_classification(input_shape, output_categories, optimizer="Nadam", given_model=None):
    if given_model and os.path.exists(given_model):
        model = load_model(given_model)
    else:
        #forward net only, we don't need transformation net because the geometric information won't change
        # ------------------------------------ Pointnet Architecture
        # input_Transformation_net
        input_points = Input(shape=input_shape)
      
        # forward net
        g = Convolution1D(64, 1, input_shape=input_shape, activation='relu')(input_points)
        g = BatchNormalization()(g)
        g = Convolution1D(64, 1, input_shape=input_shape, activation='relu')(g)
        g = BatchNormalization()(g)

        # forward net
        g = Convolution1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(128, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(1024, 1, activation='relu')(g)
        g = BatchNormalization()(g)

        # global_feature
        global_feature = MaxPooling1D(pool_size=input_shape[0])(g)

        # point_net_cls
        c = Dense(512, activation='relu')(global_feature)
        c = BatchNormalization()(c)
        c = Dropout(rate=0.7)(c)
        c = Dense(256, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Dropout(rate=0.7)(c)
        c = Flatten()(c)
        prediction = Dense(CLASS, activation='softmax')(c)
        
        model = Model(inputs=input_points, outputs=prediction)


    if GPU > 0:
        gpu_model = multi_gpu_model(model, GPU)
    else:
        gpu_model = model
 
    print(model.summary())
 
    return model, gpu_model

def build_model_segmentation(input_shape, output_categories, optimizer="Nadam", given_model=None):
    if given_model and os.path.exists(given_model):
        model = load_model(given_model)
    else:
        input_points = Input(shape=input_shape)
        g = Convolution1D(128, 1, input_shape=input_shape, activation='relu')(input_points)
        g = BatchNormalization()(g)
        g = Convolution1D(128, 1, input_shape=input_shape, activation='relu')(g)
        g = BatchNormalization()(g)

        seg_part1 = g
        g = Convolution1D(128, 1, activation='relu')(g)  
        g = BatchNormalization()(g)
        g = Convolution1D(256, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(1024, 1, activation='relu')(g)
        g = BatchNormalization()(g)

        # global_feature
        global_feature = MaxPooling1D(pool_size=input_shape[0])(g)
        global_feature = Reshape((1024,))(global_feature)
        global_feature = RepeatVector(input_shape[0])(global_feature)
        # point_net_seg
        c = concatenate([seg_part1, global_feature])
        c = Convolution1D(1024, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(512, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(256, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(1, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        # for segmentation
        prediction = Convolution1D(input_shape[0], 1, activation='sigmoid')(c)  
        model = Model(inputs=input_points, outputs=prediction)


    if GPU > 0:
        gpu_model = multi_gpu_model(model, GPU)
    else:
        gpu_model = model
 
    print(model.summary())
 
    return model, gpu_model




if __name__ == '__main__':

    path_to_train = "../input/train_1"
    start_time = time.time()
    x_train, y_train = generate_multiple_event_data(skip=0, nevents=TRAIN_EVENT)
    x_val, y_val = generate_multiple_event_data(skip=TRAIN_EVENT, nevents=VAL_EVENT)
    x_test, y_test = generate_multiple_event_data(skip=TRAIN_EVENT+VAL_EVENT, nevents=TEST_EVENT)
    
    print("------ Generating train batch takes %s seconds -------" % (time.time() - start_time))


    opt = Adam(lr=INIT_LR, decay=DECAY)
    #model_name = '2018-07-07-21-10-18.h5'
    model, gpu_model = build_model_classification(input_shape = (x_train.shape[1], x_train.shape[2]), output_categories = (y_train.shape[2]), optimizer=opt, given_model=None )
    gpu_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = gpu_model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE,epochs=EPOCHS, shuffle = True)
    
    model.save(str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))+'.h5')
    draw_train_history(history, draw_val=False)
    
    pred = gpu_model.predict(x_test)
    test_loss, test_acc = gpu_model.evaluate(x_test, y_test)
    print(test_loss, test_acc)
    
    print(pred)
