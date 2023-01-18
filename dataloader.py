import cv2
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from typing import Tuple, List
import imageio
import random
from PIL import Image
import os
import tensorflow as tf
def seed_tensorflow(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # `pip install tensorflow-determinism` first,使用与tf>2.1

#seed_tensorflow(10)
random.seed(10)

#pip install typing-extensions --upgrade
#MAX_AGE = 100

#pip install albumentations==1.1.0
def calc_normal_distribution(x, mu, sigma=1):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))


class ImageGenerator(Sequence):
    def __init__(self,pic_path, batch_size: int,df, transfomer: A.Compose,ifdigt,maxage,ifnoise,noise_rate) -> None:
        """Initialize
        Args:
            batch_size (int):

            transfomer (A.Compose):
        """
        self.pic_path=pic_path
        self.df =df
        self.batch_size = batch_size
        self.indices = np.arange(len(self.df))
        self.transformer = transfomer
        self.ifdigt=ifdigt
        self.MAX_AGE=maxage
        self.ifnoise=ifnoise
        self.noise_rate=noise_rate

        #np.random.shuffle(self.indices)

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        imgs, ages, age_dists = [], [], []
        sample_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        for _, row in self.df.iloc[sample_indices].iterrows():
            age_dist = [0] * self.MAX_AGE
            file_path = str(row["file"])
            age = int(row["age"])

            img = cv2.imread(self.pic_path+file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img=imageio.imread(self.pic_path+file_path)


            img = self.transformer(image=img)["image"]
            #img = img / 255.

            age_dist =np.array([calc_normal_distribution(i, age) for i in range(0,self.MAX_AGE)])
            #print(sum(age_dist))

            imgs.append(img)
            ages.append(age / self.MAX_AGE)
            age_dists.append(age_dist)

        age_dists=np.array(age_dists)
        ages=np.array(ages)
        imgs=np.array(imgs)
        #print(ages)
        #print(ages)

        if self.ifnoise==1:
            noise_num = int(len(ages) * self.noise_rate)
            noise_index = random.choices([x for x in range(len(ages))], k=noise_num)
            for index in noise_index:
                noise_age = ages[index]
                noise_age= random.choice([x for x in range(self.MAX_AGE) if x != noise_age]) /self.MAX_AGE
                ages[index] =noise_age
        #print(ages)
            age_dists=[]
            for a in (ages*self.MAX_AGE):
                noise_dist=np.array([calc_normal_distribution(i,a) for i in range(0, self.MAX_AGE)])
                age_dists.append(noise_dist)
                #print(sum(noise_dist))
            age_dists=np.array(age_dists)

        elif self.ifnoise==0:
            age_dists=age_dists



        if self.ifdigt==1:
            return imgs, (age_dists, ages)

        elif self.ifdigt==0:
            return imgs,ages


    def __len__(self) -> int:
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def get_test_data(test_df_path,test_pic_path):

    test_x=[]
    test_y=[]
    testdf=pd.read_csv(test_df_path,names=['file','age',''])
    transformer = A.Compose([A.Resize(height=224, width=224),
                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             ])

    for x in range(len(testdf)):
        pic=testdf.iloc[x,0]
        pic=test_pic_path+pic
        pic = cv2.imread(pic)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

        pic= transformer(image=pic)["image"]
        #pic=pic/255.


        # plt.imshow(pic)
        # plt.show()

        label=testdf.iloc[x,1]

        test_x.append(pic)
        test_y.append(label)

    test_x=np.array(test_x)

    return test_x,test_y




#
# train_generator = ImageGenerator(BATCH_SIZE,train_df, TRAIN_TRANSFORMER)

#test_x,test_y=get_test_data('./data/chalearn15/valid15_gt.csv','./data/chalearn15/Valid/')
