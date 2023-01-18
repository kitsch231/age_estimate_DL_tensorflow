import os

import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import face_recognition
#conda install -c conda-forge dlib
import cv2
#pip install numpy scipy matplotlib scikit-image scikit-learn ipython dlib
# Load the jpg file into a numpy array

files=os.listdir('./data/chalearn15/Valid/')
files=['./data/chalearn15/Valid/'+x for x in files]

for file in tqdm.tqdm(files):
    image = face_recognition.load_image_file(file)
    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    #print("I found {} face(s) in this photograph.".format(len(face_locations)))
    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        # plt.imshow(face_image)
        # plt.show()
        im = Image.fromarray(face_image)
        print(file)
        im.save(file)
