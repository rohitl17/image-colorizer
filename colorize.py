import numpy as np
import cv2

def rgb_2_gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

def dataloader(files, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    while True:
        for image in files:
            inputs.append(rgb_2_gray(cv2.resize(cv2.imread(image), (256,256))))
            targets.append(cv2.resize(cv2.imread(image), (256,256)))
            batchcount += 1
            if batchcount > batchsize:
                X = np.array(inputs, dtype='float32')
                y = np.array(targets, dtype='float32')

                X=X/255.0
                y=y/255.0


                yield (X, y)
                inputs = []
                targets = []
                batchcount = 0
            
            
def predict_dataloader(files, batchsize):
    inputs=[]
    batchcount=0
    for image in files:
        inputs.append(rgb_2_gray(cv2.resize(cv2.imread(image), (256,256))))
        batchcount += 1
        if batchcount > batchsize:
            X = np.array(inputs, dtype='float32')
            X=X/255.0

            yield (X)
            inputs = []
            batchcount = 0

def predict_on_image(file):
    X=cv2.resize(cv2.imread(image), (256,256))
    X = np.array(X, dtype='float32')
    X=X/255.0
    yield(X)