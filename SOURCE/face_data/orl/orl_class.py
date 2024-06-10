import torch
import torch.utils.data as data

class orlDataset(data.Dataset):
  def __init__(self, face_data, trans, method):
    self.face_data = face_data
    self.transform = trans
    self.method = method
  def __len__(self):
    if self.method == True:
      return len(self.face_data[0]) # self.face_data[2] is correct too
    else:
      return len(self.face_data[1]) # self.face_data[3] is correct too
  def __getitem__(self, img_dir_idx):
    if self.method == True:
      train_image = self.face_data[0][img_dir_idx]
      train_label = self.face_data[2][img_dir_idx]
      train_image_trans = self.transform(train_image)
      return train_image_trans, train_label
    else:
      test_image = self.face_data[1][img_dir_idx]
      test_label = self.face_data[3][img_dir_idx]
      test_image_trans = self.transform(test_image)
      return test_image_trans, test_label