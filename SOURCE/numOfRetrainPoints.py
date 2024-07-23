import pandas as pd

import argparse

"""
Thống kê số lượng điểm dữ liệu (ảnh khuôn mặt) đã được huấn luyện lại trong mỗi trường hợp 
(cụ thể là số lượng yêu cầu cần loại bỏ ảnh khuôn mặt) của s phân đoạn 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
args = parser.parse_args()

rp = pd.read_csv('containers/{}/numOfRetrainPoints.tmp'.format(args.container), names=['nb_retrained_points'])
print(rp['nb_retrained_points'].sum())
