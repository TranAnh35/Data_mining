# Báo Cáo Thực Hành

## Bagging
### Khái niện Bagging:
Bagging (hoặc Bootstrap aggregating) là một kiểu học tập tổng hợp trong đó nhiều mô hình cơ sở được đào tạo độc lập và song song trên các tập con khác nhau của dữ liệu huán luyện. Một tập hợp con được tạo bằng cách lấy mẫu bootstrap, trong đó các điểm dữ liệu được chọn ngẫu nhiên và thay thế.

Trong trường hợp của Bagging Classifier, dự đoán cuối cùng được đưa ra bằng cách tổng hợp các dự đoán của mô hình toàn cơ sở bằng cách sử dụng biểu quyết đa số.

Trong trường hợp của Bagging Regression, dự đoná cuối cùng được đưa ra bằng cách lấy trung bình các dự đoán của mô hình toàn cơ sở.

Bagging giúp cải thiện độ chính xác và giảm thiểu tình trạng overfitting, đặc biệt là trong các mô hình có phương sai cao.

### Cách hoạt động của Bagging Classifier:
Các bước cơ bản về cách thực hoạt động của Bagging Classifier như sau:
- Lấy mẫu Bootstrap: Trong lấy mẫu Bootstrap, 'n' tập con của tập dữ liệu training gốc được lấy mẫu ngẫu nhiên với việc thay thế. Bước này đảm bảo rằng các mô hình cơ sở được đào tạo trên các tập con đa dạng của dữ liệu, vì một số mẫu có thể xuất hiện nhiều lần trong tập con mới, trong khi một số khác có thể bị loại bỏ. Điều này giảm nguy cơ overfitting và cải thiện độ chính xác của mô hình."

```sh
Ví dụ như sau:
Tập dữ liệu training ban đầu: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Tập train được lấy mẫu 1: [2, 4, 6, 8, 8, 1, 4, 5, 3, 3]
Tập train được lấy mẫu 2: [1, 1, 1, 5, 6, 9, 4, 2, 7, 8]
Tập train được lấy mẫu 3: [10, 3, 5, 8, 5, 7, 8, 4, 5, 6]
```
- Đào tạo mô hình cơ sở: Trong mô hình Bagging, nhiều mô hình được sử dụng. Sau khi thực hiện Bootstrap Sampling, mỗi mô hình cơ bản được đào tạo độc lập bằng một thuật toán cụ thể như Decision Tree, Support vector machines hoặc Logistc Regression trên một tập con dữ liệu được lấy mẫu theo Bootstrap khác nhau. Những mô hình như vậy thường được gọi là "Weak learners" hay "Học yếu" vì chúng có thể không chính xác cao khi chúng đứng một mình. Do đó mỗi cơ bản được đào tạo độc lập trên các tập cọn khác nhau của dữ liệu, để làm cho mô hình hiệu quả về mặt tính toán và tiết kiệm thời gian, các mô hình cơ bản có thể được đào tạo song song.

- Tổng hợp: Sau khi tất cả các mô hình cơ bản được đào tạo, chúng được sử dụng để đưa ra dự đoán trên tập dữ liệu chưa nhìn thấy, tức là tập con dữ liệu mà mô hình cơ bản đó không được đào tạo. Trong Bagging Classifier, nhãn lớp dự đoán cho một ví dụ cụ thể được chọn dựa trên phương pháp bỏ phiếu đa số. Lớp có số phiếu nhiều nhất sẽ là dự đoán của mô hình.

- Đánh giá Out-Of-Bag (OOB): Một số mẫu bị loại bỏ khỏi tập training con của các mô hình cơ sở cụ thể trong quá trình Bootstrap. Những mẫu "out-of-bag" này có thể được sử dụng để ước lượng hiệu suất của mô hình mà không cần đến quá trình kiểm tra chéo

- Dự đoán cuối cùng: Sau khi tổng hợp các dự đoán từ tất cả các mo hình cơ sổ, Bagging tạo ra một dự đoán cuối cùng cho mỗi ví dụ.

### Thuật toán của Bagging Classifier:

```sh
Quá trình tạo bộ phân loại:
Giả sử N là kích thước của tập training.
Với mỗi lần trong số t lần lặp:
    Lấy mẫu N ví dụ với việc thay thế từ tập train gốc.
    Áp dụng thuật toán học trên mẫu.
    Lưu trữ bộ phân loại kết quả.
```

```sh
Quá trình phân loại:
Đối với mỗi lần trong t lần lặp:
    Dự đoán lớp của vs dụ bằng cách sự dụng bộ phân loại.
    Trả về lớp được dự đoán nhiều nhất.
```