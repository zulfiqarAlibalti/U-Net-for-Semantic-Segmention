from Architecture import multi_class_unet_architecture,jacard,jacard_loss
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.io import imshow
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

# %%%%%%%%%%%%%%%%%%%% Training the dataset%%%%%%%%%%%%%%
train_path = r"original_images/*.jpg"
mask_path = r"image_masks/*.png"
def importing_data(path):
    sample = []
    for filename in glob.glob(path):
        img  = Image.open(filename,'r')
        img = img.resize((256,256))
        img = np.array(img)
        sample.append(img)
    return sample

train_data = importing_data(train_path)
train_data = np.asarray(train_data)
# print(train_data[1])
mask_data = importing_data(mask_path)
mask_data = np.asarray(mask_data)

# %%%%%%%%%%%%%%% Visualization %%%%%%%%%%%%%%%%%%%%%%
x = random.randint(0, len(train_data))
plt.figure(figsize=(24,18))
plt.subplot(1,2,1)
imshow(train_data[x])
plt.subplot(1,2,2)
imshow(mask_data[x])
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%
scaler = MinMaxScaler()
nsamples , nx,ny, nz = train_data.shape
d2_train_data = train_data.reshape((nsamples,nx*ny*nz))
train_images = scaler.fit_transform(d2_train_data)
train_images = train_images.reshape(400,256,256,3)

# %%%%%%%%%%%%%%%%%% Label the mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels = pd.read_csv(r"class_dict_seg.csv")
labels = labels.drop(['name'],axis = 1)
labels = np.array(labels)

def image_labels(label):
    image_labels = np.zeros(label.shape ,dtype=np.uint8)
    for i in range(24):
        image_labels[np.all(label==labels[i,:],axis=-1)]=i
    image_labels = image_labels[:,:,0]
    return image_labels


label_final = []
for i in range(mask_data.shape[0]):
    label = image_labels(mask_data[i])
    label_final.append(label)

label_final = np.array(label_final)

# %%%%%%%%%%%%%%%%%%%%%%%% Train Test Separation of Data%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_classes = len(np.unique(label_final))
labels_cat = to_categorical(label_final, num_classes=n_classes)
train_x, test_x, train_y, test_y = train_test_split(train_images,labels_cat,test_size=0.25,random_state=42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%% Unet Training Part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img_h = train_x.shape[1]
img_w = train_x.shape[2]
img_c = train_x.shape[3]

metrics = ['accuracy', jacard]

def get_model():
    return multi_class_unet_architecture(n_classes = n_classes,height=img_h,width=img_w,channels=img_c)

model = get_model()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = metrics)
model.summary()

history = model.fit(train_x,train_y,
                    batch_size=16,verbose=1,epochs=100,validation_data=(test_x,test_y),shuffle=False)

# %%%%%%%%%%%%%%%%%%%%%%% Loss and Accuracy Plot of model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss  = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1)
plt.plot(epochs,loss,'y',label = 'Training loss')
plt.plot(epochs,val_loss,label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc  = history.history['jacard']
val_acc = history.history['val_jacard']
epochs = range(1, len(loss) +1)
plt.plot(epochs,acc,'y',label = 'Training accuracy')
plt.plot(epochs,val_acc,label = 'Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred  = model.predict(test_x)
y_pred_argmax = np.argmax(y_pred, axis = 3)
y_test_argmax = np.argmax(test_y, axis = 3)

test_jacard = jacard(test_y, y_pred)
print(test_jacard)

fig, ax = plt.subplot(5,3,figsize = (12,18))
for i in range(0,5):
    test_img_number = random.randint(0,len(test_x))
    test_img = test_x[test_img_number]
    ground_truth = y_test_argmax[test_img_number]
    test_img_input = np.expand_dims(test_img,0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis =3)[0,:,:]

    ax[i,0].imshow(test_img)
    ax[i,0].set_title("RGB Image", frontsize = 16)
    ax[i,1] = imshow(ground_truth)
    ax[i,1] = plt.title("Ground Truth",fontsize = 16)
    ax[i, 2] = imshow(predicted_img)
    ax[i, 2] = plt.title("Prediction", fontsize=16)
    i+= i

plt.show()
