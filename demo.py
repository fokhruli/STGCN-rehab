import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('inputs', 'Data/input.csv', 'Testing')
flags.DEFINE_string('labels', 'Data/label.csv', 'labeling')

index_Spine_Base=0
index_Spine_Mid=4
index_Neck=8
index_Head=12   # no orientation
index_Shoulder_Left=16
index_Elbow_Left=20
index_Wrist_Left=24
index_Hand_Left=28
index_Shoulder_Right=32
index_Elbow_Right=36
index_Wrist_Right=40
index_Hand_Right=44
index_Hip_Left=48
index_Knee_Left=52
index_Ankle_Left=56
index_Foot_Left=60  # no orientation    
index_Hip_Right=64
index_Knee_Right=68
index_Ankle_Right=72
index_Foot_Right=76   # no orientation
index_Spine_Shoulder=80
index_Tip_Left=84     # no orientation
index_Thumb_Left=88   # no orientation
index_Tip_Right=92    # no orientation
index_Thumb_Right=96  # no orientation

body_part = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]

def Demo(inputs, labels):
    """Load the data"""
    x = pd.read_csv(inputs, header = None).iloc[:,:].values
    label = pd.read_csv(labels, header = None).iloc[:,:].values
    batch_size = label.shape[0]
    num_timestep = x.shape[0]
    num_channel = 3
    num_joints = 25
    """Load the scaling object"""
    sc_x = joblib.load("Data/sc_x.save") 
    sc_y = joblib.load("Data/sc_y.save") 
    
    X = np.zeros((x.shape[0],num_joints*num_channel)).astype('float32')
    for row in range(x.shape[0]):
        counter = 0
        for parts in body_part:
            for i in range(num_channel):
                X[row, counter+i] = x[row, parts+i]
            counter += num_channel 
    
    X = sc_x.transform(X)         
    
    X_ = np.zeros((batch_size, num_timestep, num_joints, num_channel))
    
    for batch in range(X_.shape[0]):
        for timestep in range(X_.shape[1]):
            for node in range(X_.shape[2]):
                for channel in range(X_.shape[3]):
                    X_[batch,timestep,node,channel] = X[timestep+(batch*num_timestep),channel+(node*num_channel)]
               
    X = X_
    
    """Load the model and weights"""
    with open('pretrain model/rehabilitation.json', 'r') as f:
        model_json = f.read()
    
    model = tf.keras.models.model_from_json(model_json, custom_objects={'tf': tf})
    model.load_weights("pretrain model/best_model.hdf5")
    
    y_pred = model.predict(X)
    y_pred = sc_y.inverse_transform(y_pred) 
    print("predicted value:", y_pred[0,0])
    #print()
    print("actual value:", label[0,0])       

Demo(FLAGS.inputs, FLAGS.labels)         