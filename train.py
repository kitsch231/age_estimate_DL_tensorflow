import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop, Adam,SGD
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import categorical_accuracy,binary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow import keras
from dataloader import ImageGenerator,get_test_data
import albumentations as A
from my_model import *
import os
import random
def seed_tensorflow(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # `pip install tensorflow-determinism` first,使用与tf>2.1

# seed_tensorflow(10)

# 可视化方法
#tensorboard --logdir=D:\A_sell_project\cv\DL_resnet\logs\dl_fgnet --host=127.0.0.1

#
model_type='nodl' #dl为标签分布学习，nodl为普通回归
train_model='all'#train,test,all可选，分表表示训练和测试模式,以及训练并测试三种模式
dataset='fgnet'#数据集选取，chalearn15,chalearn16,utkface,fgnet四个数据可用
ifnoise=1#是否添加噪声标签1为添加 0为不添加
noise_rate=0.2#噪声标签的比例,范围0-1
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 1e-3
BATCH_SIZE =32#批大小
epoch=100



if dataset=='chalearn15':
    train_df_path='./data/chalearn15/Train15_gt.csv'
    test_df_path='./data/chalearn15/valid15_gt.csv'

    train_pic_path='./data/chalearn15/Train/'
    test_pic_path='./data/chalearn15/Valid/'
    maxage=100#最大年龄

elif dataset=='chalearn16':
    train_df_path='./data/chalearn16/train16_gt.csv'
    test_df_path='./data/chalearn16/valid16_gt.csv'

    train_pic_path='./data/chalearn16/Train/'
    test_pic_path='./data/chalearn16/Valid/'
    maxage=100#最大年龄

elif dataset=='fgnet':
    train_df_path='./data/fgnet/train.csv'
    test_df_path='./data/fgnet/val.csv'

    train_pic_path='./data/fgnet/images/'
    test_pic_path='./data/fgnet/images/'
    maxage=100

elif dataset=='utkface':
    train_df_path='./data/utkface/train.csv'
    test_df_path='./data/utkface/val.csv'

    train_pic_path='./data/utkface/images/'
    test_pic_path='./data/utkface/images/'
    maxage=116

if model_type=='dl':
    model=dl_model(maxage=maxage)
    ifdigt =1
    loss=[tf.keras.losses.kullback_leibler_divergence,
            tf.keras.losses.mean_absolute_error]#dl模式双loss
    monitor = 'val_pred_age_loss'
    loss_weight = [1, 20]

elif model_type=='nodl':
    model=nodl_model(maxage=maxage)
    ifdigt = 0
    loss=tf.keras.losses.mean_absolute_error#nodl模式单loss
    loss_weight=None
    monitor = 'val_loss'

if ifnoise==1:
    filepath = 'noise_'+str(noise_rate)+'_'+model_type + '_' + dataset  # 模型及loss结果等保存路径
elif ifnoise==0:
    filepath =model_type+'_'+dataset#模型及loss结果等保存路径


if train_model=='train' or train_model=='all':
    df=pd.read_csv(train_df_path,names=['file','age',''])
    valdf=df.sample(frac=0.2,random_state=10)
    traindf=df.drop(index=valdf.index.to_list())


    #训练集数据增强
    train_TRANSFORMER = A.Compose(
        [ A.Resize(height=INPUT_SHAPE[1], width=INPUT_SHAPE[0]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=1.0),
            A.CenterCrop(p=1.0, height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    #测试集数据增强
    val_TRANSFORMER=A.Compose([A.Resize(height=INPUT_SHAPE[1], width=INPUT_SHAPE[0]),
                               A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                               ])

    train_generator = ImageGenerator(train_pic_path,BATCH_SIZE,traindf, train_TRANSFORMER,ifdigt=ifdigt,maxage=maxage,
                                     ifnoise=ifnoise,noise_rate=noise_rate)
    val_generator=ImageGenerator(train_pic_path,BATCH_SIZE,valdf,val_TRANSFORMER,ifdigt=ifdigt,maxage=maxage,
                                 ifnoise=ifnoise,noise_rate=noise_rate)

    tbCallBack = TensorBoard(log_dir="./logs/"+filepath)
    #保存最优模型
    checkpoint = ModelCheckpoint('./model/'+filepath, monitor=monitor, verbose=1, save_best_only=True, period=1)
    #optimizer = Adam(learning_rate=LEARNING_RATE,clipnorm=1.0,epsilon=1e-8)
    optimizer=SGD(momentum=0.9,nesterov=True)

    model.compile(optimizer=optimizer,loss= loss,loss_weights=loss_weight)
    print(model.summary())
    history=model.fit_generator(train_generator,epochs=epoch,validation_data=val_generator,callbacks=[checkpoint,tbCallBack])
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title(filepath, fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('epoch', fontsize='10')
    plt.legend()
    plt.savefig('./loss/'+filepath+'.png')
    #plt.show()

    if model_type=='dl':
        plt.figure()
        plt.plot(history.history['pred_age_loss'], label='train')
        plt.plot(history.history['val_pred_age_loss'], label='val')
        plt.title(filepath, fontsize='12')
        plt.ylabel('mae', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.savefig('./loss/' + filepath + '_mae.png')
        #plt.show()

    if model_type == 'nodl':
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title(filepath, fontsize='12')
        plt.ylabel('mae', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.savefig('./loss/' + filepath + '_mae.png')
        #plt.show()

    result=pd.DataFrame()
    result['loss']=history.history['loss']
    result['val_loss']=history.history['val_loss']
    #add row to end of DataFrame
    result.to_csv('./loss/'+filepath+'.csv',index=None)


if train_model=='test' or train_model=='all':
    print('test...')
    model=load_model('./model/'+filepath)
    test_x,test_y=get_test_data(test_df_path,test_pic_path)


    if model_type=='dl':
        pre=model.predict(test_x)[1]*maxage
        pre=[x[0] for x in pre]

    elif model_type=='nodl':
        pre=model.predict(test_x)*maxage
        pre=[x[0] for x in pre]

    print(pre)
    print(test_y)
    from sklearn.metrics import mean_absolute_error # 平方绝对误差

    mae=mean_absolute_error(test_y,pre)
    print(filepath+'_mae:',mae)

    #add row to end of DataFrame
    log_dict={'name':filepath,'mae':mae,'epoch':epoch,'batch_size':BATCH_SIZE}
    log=pd.DataFrame(log_dict,index=[0]).T
    log.to_csv('./loss/'+filepath+'_mae.csv')