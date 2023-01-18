import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from typing import Tuple, List
import imageio
import random
from PIL import Image
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import albumentations as A
import cv2
import tqdm

from tensorflow.keras import models as ksmodels
import itertools as it
import face_recognition

def get_test_data(pic_path):
    test_x=[]
    for pic in tqdm.tqdm(pic_path):
        transformer = A.Compose([A.Resize(height=224, width=224),
                                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                 ])

        pic = cv2.imread(pic)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

        # pic = face_recognition.load_image_file(pic)
        # face_locations = face_recognition.face_locations(pic, number_of_times_to_upsample=0, model="cnn")
        # for face_location in face_locations:
        #     # Print the location of each face in this image
        #     top, right, bottom, left = face_location
        #     pic= pic[top:bottom, left:right]
        # plt.imshow(pic)
        # plt.show()


        pic= transformer(image=pic)["image"]
        test_x.append(pic)

        # plt.imshow(pic)
        # plt.show()
        # print(pic)
    test_x=np.array(test_x)
    return test_x

def use_model(model_path,pic_dir):


    model_path=model_path
    dir=pic_dir


    args=model_path.split('/')[-2]
    args=args.split('_')

    if 'utkface' in args:
        maxage=116
    else:
        maxage=100

    if 'dl' in args:
        ifdl=1
    else:
        ifdl=0

    print('**********test***********:',model_path,'dl：{}'.format(ifdl),'maxage:{}'.format(maxage))
    pics=os.listdir(dir)
    pics=[dir+'/'+x for x in pics]


    test_x=get_test_data(pics)
    model=ksmodels.load_model(model_path)

    if ifdl==1:
        pre=model.predict(test_x)[1]*maxage
        pre=[x[0] for x in pre]
    elif ifdl==0:
        pre=model.predict(test_x)*maxage
        pre=[x[0] for x in pre]
    #
    df_path=pic_dir.split('/')[-2]+'.csv'
    model_name=model_path.split('/')[-2]
    print(df_path)
    if os.path.exists(df_path):
        df=pd.read_csv(df_path)
    else:
        df=pd.DataFrame()
    df['files']=pics
    df[model_name]=pre
    print(df)
    df.to_csv(df_path,index=None)

dirs=os.listdir('NPR_test')
dirs=['./NPR_test/'+x+'/' for x in dirs]
print(dirs)

models=os.listdir('model')
models=['./model/'+x+'/' for x in models]
print(models)

for e in it.product(models,(dirs[0],)):
    #print(e)
    model_path,pic_dir=e
    use_model(model_path,pic_dir)


for e in it.product(models,(dirs[1],)):

    model_path,pic_dir=e
    use_model(model_path,pic_dir)

for e in it.product(models,(dirs[2],)):

    model_path,pic_dir=e
    use_model(model_path,pic_dir)