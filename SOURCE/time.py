import pandas as pd

import argparse

# Thống kê thời gian huấn luyện lại trong trường hợp dữ liệu chia thành s phân đoạn 

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
args = parser.parse_args()

time_csv = pd.read_csv('containers/{}/times/times.tmp'.format(args.container), names=['time'])
print('{},{}'.format(time_csv['time'].sum(),time_csv['time'].mean()))