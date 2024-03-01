# Báo Cáo Thực Hành

## Bagging
### Khái niện Bagging
Bagging (hoặc Bootstrap aggregating) là một kiểu học tập tổng hợp trong đó nhiều mô hình cơ sở được đào tạo độc lập và song song trên các tập con khác nhau của dữ liệu huán luyện. Một tập hợp con được tạo bằng cách lấy mẫu bootstrap, trong đó các điểm dữ liệu được chọn ngẫu nhiên và thay thế.
Trong trường hợp của Bagging Classifier, dự đoán cuối cùng được đưa ra bằng cách tổng hợp các dự đoán của mô hình toàn cơ sở bằng cách sử dụng biểu quyết đa số.ư
Trong trường hợp của Bagging Regression, dự đoná cuối cùng được đưa ra bằng cách lấy trung bình các dự đoán của mô hình toàn cơ sở.
Bagging giúp cải thiện độ chính xác và giảm thiểu tình trạng overfitting, đặc biệt là trong các mô hình có phương sai cao.

### Cách hoạt động của Bagging Classifier
Các bước cơ bản về cách thực hoạt động của Bagging Classifier như sau:
- Lấy mẫu Bootstrap:
