import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from ar_class import arDataset
import json

dataset_path = '../../../DATASET/ar'
images = []
labels = []

# Duyệt qua các tập tin trong thư mục
for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg'):
        # Read face images
        image = cv2.imread(os.path.join(dataset_path, filename))
        images.append(image)
        
        # Lấy ID từ tên tệp và gán làm nhãn # Extract ID from file name and assign to label
        id_num = int(filename.split('-')[1])
        if filename.startswith('M'):
            label = id_num - 1
        elif filename.startswith('W'):
            label = id_num + 49
        labels.append(label)

# Convert normal array to numpy array
images = np.array(images, dtype=np.uint8)
labels = np.array(labels)

n_class = np.max(labels) + 1 

X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
for i in range(np.max(labels) + 1):
    idx = np.where(labels == i)[0]
    img_tmp, label_tmp = images[idx], labels[idx]
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(img_tmp, label_tmp, test_size=0.2)
    X_train,X_test, y_train, y_test = np.vstack([X_train_tmp, X_train]), 
    np.vstack([X_test_tmp, X_test]), 
    np.vstack([y_train_tmp, y_train]), 
    np.vstack([y_test_tmp, y_test])

train_trans = transforms.Compose([
        transforms.RandomResizedCrop(size = (227,227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229, 0.224, 0.225])
    ])

test_trans = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229, 0.224, 0.225])
    ])

face_data = X_train, X_test, y_train, y_test
train_data = arDataset(face_data, train_trans, isTrain = True)
test_data = arDataset(face_data, test_trans, isTrain = False)


if not os.path.exists('ar_info'):
    ar_info = {
        "nb_train": train_data.__len__(),
        "nb_test": test_data.__len__(),
        "input_shape": np.array(train_data.__getitem__(0)[0].numpy().shape, dtype=int).tolist(),
        "nb_classes": n_class,
        "dataloader": "dataloader"
    }
    
    with open("ar_info", "w") as orl_info_file:
        json.dump(ar_info, orl_info_file)

X_train = np.array([train_data.__getitem__(i)[0].numpy() for i in range(train_data.__len__())], dtype="float32")
y_train = np.array([train_data.__getitem__(i)[1] for i in range(train_data.__len__())])
X_test = np.array([test_data.__getitem__(i)[0].numpy() for i in range(test_data.__len__())], dtype="float32")
y_test = np.array([test_data.__getitem__(i)[1] for i in range(test_data.__len__())])

if not os.path.exists(f'ar_{n_class}.npy'):
    np.save(f'ar{n_class}_data.npy', {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})