import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from orl_class import orlDataset
import json

n_class = len(os.listdir('../../../DATASET/orl'))

# (Read and prepare) image and create labels
images = []
labels = []

for i in range(n_class):
    for j in range(10):
        image = cv2.imread('../../../DATASET/orl/s{}/{}.pgm'.format(i+1, j+1), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        images.append(image)
        labels.append(i)

images = np.array(images, dtype=np.uint8)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(size=(92,92)),
    #transforms.RandomRotation((0,15)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5622, 0.5622, 0.5622], 
                         std=[0.1860, 0.1860, 0.1860])
])

test_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(size=(92,92)),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5622, 0.5622, 0.5622], 
                         std=[0.1860, 0.1860, 0.1860])
])

face_data = X_train, X_test, y_train, y_test
train_data = orlDataset(face_data, train_trans, isTrain = True)
test_data = orlDataset(face_data, test_trans, isTrain = False)


if not os.path.exists('orl_info'):
    orl_info = {
        "nb_train": train_data.__len__(),
        "nb_test": test_data.__len__(),
        "input_shape": np.array(train_data.__getitem__(0)[0].numpy().shape, dtype=int).tolist(),
        "nb_classes": n_class,
        "dataloader": "dataloader"
    }
    
    with open("orl_info", "w") as orl_info_file:
        json.dump(orl_info, orl_info_file)

X_train = np.array([train_data.__getitem__(i)[0].numpy() for i in range(train_data.__len__())], dtype="float32")
y_train = np.array([train_data.__getitem__(i)[1] for i in range(train_data.__len__())])
X_test = np.array([test_data.__getitem__(i)[0].numpy() for i in range(test_data.__len__())], dtype="float32")
y_test = np.array([test_data.__getitem__(i)[1] for i in range(test_data.__len__())])

if not os.path.exists(f'orl_{n_class}.npy'):
    np.save(f'orl{n_class}_data.npy', {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})
