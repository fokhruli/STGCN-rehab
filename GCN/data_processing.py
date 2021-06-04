import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from IPython.core.debugger import set_trace

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

class Data_Loader():
    def __init__(self, dir):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]
        return body_parts
    
    def import_dataset(self):
        train_x = pd.read_csv(self.dir+"/Train_X.csv", header = None).iloc[:,:].values
        train_y = pd.read_csv(self.dir+"/Train_Y.csv", header = None).iloc[:,:].values
        return train_x, train_y
            
    def preprocessing(self):
        X_train = np.zeros((self.train_x.shape[0],self.num_joints*self.num_channel)).astype('float32')
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter+i] = self.train_x[row, parts+i]
                counter += self.num_channel 
        
        y_train = np.reshape(self.train_y,(-1,1))
        X_train = self.sc1.fit_transform(X_train)         
        y_train = self.sc2.fit_transform(y_train)
        
        X_train_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))
        
        for batch in range(X_train_.shape[0]):
            for timestep in range(X_train_.shape[1]):
                for node in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*self.num_timestep),channel+(node*self.num_channel)]
        
                        
        X_train = X_train_                
        return X_train, y_train