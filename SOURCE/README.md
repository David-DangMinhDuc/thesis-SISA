# Nâng cao hiệu quả các mô hình nhận dạng khuôn mặt dùng SISA

## Thông tin đề tài

- ***Tên đề tài***: Nâng cao hiệu quả các mô hình nhận dạng khuôn mặt dùng SISA
- ***Loại đề tài***: Khóa luận tốt nghiệp
- ***Giảng viên hướng dẫn***: PGS. TS. Lê Hoàng Thái
- ***Giảng viên phản biện***: TS. Bùi Tiến Lên
- ***Sinh viên thực hiện***:

| ***Họ và tên***  | ***MSSV***  | ***Chuyên ngành***  |
|    ---     | ---   | ---   |
| Bùi Tiến Đạt  | 20127130  | Khoa học máy tính  |
| Đặng Minh Đức  | 20127136  | Khoa học máy tính  |

- ***Ngày bảo vệ - Hội động bảo vệ***: 17/08/2024 - Hội đồng Khoa học máy tính 3
- ***Địa điểm bảo vệ***: Phòng I44 - Trường Đại học Khoa học tự nhiên, ĐHQG-HCM

## Cách chạy code trong đề tài

- Trước hết, chúng tôi sẽ chạy code trên file `face_data/orl/prepare_orl.py` hoặc `face_data/ar/prepare_ar.py` nhằm xuất ra dữ liệu sau khi chia thành tỉ lệ tùy thuộc vào tập dữ liệu (ORL: 80% train và 20% test, AR: 80% train và 20% test) phục vụ cho `dataloader.py`.
- Tiếp đến, chúng tôi tiến hành thực nghiệm trên đề tài này theo từng bước sau:
    * Phân chia dữ liệu thành `s` phân đoạn cụ thể, khởi tạo số lượng yêu cầu loại bỏ dựa vào phân phối xác suất (trong đề tài này chúng tôi chỉ áp dụng SISA vào các mô hình nhận dạng khuôn mặt AlexNet và VGG-16 theo phân phối xác suất đều ("uniform")) bằng cách chạy file `bash orl_init.sh s` hoặc `bash ar_init.sh s`.
    * Huấn luyện phương pháp SISA với `s` phân đoạn, `r` lát cắt và `e` kỷ nguyên cụ thể trong "container": `bash orl_train.sh s r e` hoặc `bash ar_train.sh s r e`.
    * Kiểm thử `s` shard cụ thể: `bash orl_predict.sh s` hoặc `bash ar_predict.sh s`.
    * Tổng hợp các tham số đầu ra của từng phân đoạn và xuất ra kết quả trong file CSV: `bash orl_data.sh s r e` hoặc `bash ar_data.sh s r e` với `s` là số phân đoạn, `r` là số lát cắt và `e` là số kỷ nguyên. Nếu muốn đưa ra kết quả trong trường hợp `baseline` (tức là một phần của phân đoạn), chúng tôi sẽ chạy các file `bash orl_data_baseline.sh s r e` hoặc `bash ar_data_baseline.sh s r e`. Đối với bước này, chúng tôi muốn đưa ra thông tin về `s`, `r`, `e` nhằm phục vụ cho việc trực quan hóa dữ liệu nói riêng và báo cáo khóa luận nói chung. 

## Kết quả thực nghiệm

### 1. Kết quả thực nghiệm dựa trên số phân đoạn
#### a. Thời gian huấn luyện và độ chính xác
![Biểu đồ thể hiện thời gian huấn luyện lại (thời gian phân tích) và độ chính xác dựa trên phân đoạn của hai tập dữ liệu AR Face Database và ORL](./vis_img/plot_base_shards.png)

#### b. Số lượng ảnh khuôn mặt được huấn luyện lại
![Biểu đồ thể hiện số lượng ảnh khuôn mặt được huấn luyện lại dựa trên phân đoạn của hai tập dữ liệu AR Face Database và ORL](./vis_img/plot_ret_pts.png)

### 2. Kết quả thực nghiệm dựa trên số lát cắt
![Biểu đồ thể hiện độ chính xác dựa trên số lát cắt của hai tập dữ liệu AR Face Database và ORL](./vis_img/plot_base_slices.png)

### 3. Tốc độ huấn luyện
#### 3.1. Theo số phân đoạn
##### 3.1.1. Tập dữ liệu AR Face Database

| Trường hợp  | Thời gian huấn luyện lại trung bình (s)  | Tốc độ huấn luyện lại trung bình |
|    ---     | ---   | ---   |
| Huấn luyện lại từ đầu (s = 1, r = 1)  | 14.93  | 1  |
| s = 5, r = 1  | 11.75  | 1.59  |
| s = 10, r = 1  | 17.34  | 1.24  |

với s là số phân đoạn và r là số lát cắt
##### 3.1.2. Tập dữ liệu ORL

| Trường hợp  | Thời gian huấn luyện lại trung bình (s) | Tốc độ huấn luyện lại trung bình |
|    ---     | ---   | ---   |
| Huấn luyện lại từ đầu (s = 1, r = 1)  | 13.73  | 1  |
| s = 5, r = 1  | 13.11  | 1.12  |
| s = 10, r = 1  | 14.42  | 1.02  |

với s là số phân đoạn và r là số lát cắt

#### 3.2. Theo số lát cắt
##### 3.2.1. Tập dữ liệu AR Face Database

| Trường hợp  | Thời gian huấn luyện lại trung bình (s)  | Tốc độ huấn luyện lại trung bình |
|    ---     | ---   | ---   |
| Huấn luyện lại từ đầu (s = 1, r = 1)  | 6.23  | 1  |
| s = 5, r = 1  | 4.56  | 1.85  |
| s = 5, r = 2  | 3.48  | 2.78  |

với s là số phân đoạn, r là số lát cắt và epoch là 10

##### 3.2.2. Tập dữ liệu ORL

| Trường hợp  | Thời gian huấn luyện lại trung bình (s) | Tốc độ huấn luyện lại trung bình |
|    ---     | ---   | ---   |
| Huấn luyện lại từ đầu (s = 1, r = 1)  | 13.73  | 1  |
| s = 5, r = 1  | 13.11  | 1.12  |
| s = 5, r = 2  | 7.75  | 1.83  |
| s = 5, r = 3  | 6.78  | 2.04  |

với s là số phân đoạn, r là số lát cắt và epoch là 30

## Nguồn tham khảo
- Các bài báo tham khảo được trình bày trong báo cáo
- Code tham khảo: https://github.com/cleverhans-lab/machine-unlearning