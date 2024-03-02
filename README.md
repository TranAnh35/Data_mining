# Báo Cáo Thực Hành

## Bagging
### Khái niện Bagging:
Bagging (hoặc Bootstrap aggregating) là một kiểu học tập tổng hợp trong đó nhiều mô hình cơ sở được đào tạo độc lập và song song trên các tập con khác nhau của dữ liệu huán luyện. Một tập hợp con được tạo bằng cách lấy mẫu bootstrap, trong đó các điểm dữ liệu được chọn ngẫu nhiên và thay thế.

Trong trường hợp của Bagging Classifier, dự đoán cuối cùng được đưa ra bằng cách tổng hợp các dự đoán của mô hình toàn cơ sở bằng cách sử dụng biểu quyết đa số.

Trong trường hợp của Bagging Regression, dự đoná cuối cùng được đưa ra bằng cách lấy trung bình các dự đoán của mô hình toàn cơ sở.

Bagging giúp cải thiện độ chính xác và giảm thiểu tình trạng overfitting, đặc biệt là trong các mô hình có phương sai cao.

![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/image3_78e8da325b.png)

(Nguồn: https://www.datacamp.com/tutorial/what-bagging-in-machine-learning-a-guide-with-examples)

### Cách hoạt động của Bagging Classifier:
Các bước cơ bản về cách thực hoạt động của Bagging Classifier như sau:
- Lấy mẫu Bootstrap: Trong lấy mẫu Bootstrap, 'n' tập con của tập dữ liệu training gốc được lấy mẫu ngẫu nhiên với việc thay thế. Bước này đảm bảo rằng các mô hình cơ sở được đào tạo trên các tập con đa dạng của dữ liệu, vì một số mẫu có thể xuất hiện nhiều lần trong tập con mới, trong khi một số khác có thể bị loại bỏ. Điều này giảm nguy cơ overfitting và cải thiện độ chính xác của mô hình.

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

- Dự đoán cuối cùng: Sau khi tổng hợp các dự đoán từ tất cả các mô hình cơ sổ, Bagging tạo ra một dự đoán cuối cùng cho mỗi ví dụ.

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

### Xây dựng thuật toán Bagging Classifier:
Ở phần này, ta xây dựng code Bagging. Code được viết bằng ngôn ngữ Python với class BaggingClassifier.

Class BaggingClassifier gồm các tham số đầu vào:
- base_classifier: Là mô hình cơ sở của thuật toán.
- n_estimators: Là số lượng các base_classifier được sử dụng để huấn luyện.
- classifiers: Là một list trống để lưu trữ các base_classifier đã được huấn luyện.
- history: Là một dictionary được sử dụng để theo dõi các matric trong quá trình huấn luyện:
    - accuracy: Lưu trữ giá trị độ chính xác trên tập train sau mỗi một estimator.
    - loss: Lưu trữ giá trị của hàm mất mát trên tập train sau mỗi một estimator.
    - val_accuracy: Lưu trữ giá trị độ chính xác trên tập validation sau mỗi một estimator.
    - val_loss: Lưu trữ giá trị của hàm mất mát trên tập validation sau mỗi một estimator

Bên cạnh đó, trong class BaggingClassifier cí các method:
- _bootstrap_sample: Thực hiện lấy mẫu bootstrap có thay thế bằng cách chọn ngẫu nhiên các chỉ số có thay thế.
- fit: Thực hiện huấn luyện mô hình và theo dõi hiệu suất.
- predict: Trả về kết quả dự đoán.

### Ưu điểm của Bagging:
- Cải thiện độ chính xác (chống overfitting): Kỹ thuật lấy mẫu tái chọn (bootstrap) trong bagging giúp giảm thiểu overfitting bằng cách giảm sự phụ thuộc quá mức vào dữ liệu huấn luyện. Các mô hình con được huấn luyện trên các tập dữ liệu khác nhau, giúp mô hình tổng thể trở nên ít nhạy cảm hơn với các biến động trong dữ liệu huấn luyện.
- Giảm thiểu sai lệch (giảm phương sai): Bagging giúp giảm phương sai của mô hình bằng cách sử dụng nhiều mô hình con có độ đa dạng cao. Khi kết hợp các dự đoán từ các mô hình con này, phương sai của dự đoán cuối cùng thường thấp hơn so với một mô hình đơn lẻ.
- Khả năng xử lý dữ liệu nhiễu: Khi kết hợp nhiều dự đoán từ các mô hình con, bagging tạo ra dự đoán tổng thể có độ chính xác và độ tin cậy cao hơn so với một mô hình đơn lẻ. Điều này làm cho bagging trở thành một phương pháp mạnh mẽ cho các bài toán dự đoán.
- Dễ dàng thực hiện: bagging là một kỹ thuật đơn giản và dễ triển khai, có thể được áp dụng cho nhiều loại mô hình học máy khác nhau mà không cần thay đổi cấu trúc của chúng.
- Tăng cường tính ổn định: Bagging giúp cải thiện tính ổn định của mô hình bằng cách giảm thiểu tác động của dữ liệu nhiễu hoặc biến động. Điều này làm cho mô hình trở nên ít nhạy cảm hơn đối với thay đổi nhỏ trong dữ liệu.

Bagging là một kỹ thuật học máy hiệu quả có thể mang lại nhiều lợi ích bao gồm cải thiện độ chính xác, giảm thiểu sai lệch, tăng cường khả năng xử lý dữ liệu nhiễu và dễ dàng thực hiện. Tuy nhiên, bagging cũng có một số nhược điểm sẽ được nói rõ hơn ở mục dưới đây.

### Nhược điểm của Bagging:
- Tốn thời gian và tài nguyên tính toán: Khi kết hợp nhiều dự đoán từ các mô hình con, bagging tạo ra dự đoán tổng thể có độ chính xác và độ tin cậy cao hơn so với một mô hình đơn lẻ. Điều này làm cho bagging trở thành một phương pháp mạnh mẽ cho các bài toán dự đoán.
- Giảm độ tin cậy (Không giảm bias): Bagging thường tập trung vào việc giảm phương sai (variance) mà ít quan tâm đến bias (độ chệch) của mô hình. Do đó, nếu mô hình cơ bản có bias cao, bagging có thể không giúp cải thiện hiệu suất của mô hình đáng kể.
- Khó khăn trong việc chọn mô hình: Trong một số trường hợp, bagging có thể không cải thiện hiệu suất của mô hình, đặc biệt là khi mô hình cơ bản đã có tính nhất quán cao đối với dữ liệu. Trong trường hợp này, việc thêm các mô hình con có thể không cần thiết và chỉ tăng thêm chi phí tính toán.
- Có thể không hiệu quả với một số tập dữ liệu: Bagging thường cần điều chỉnh các tham số như số lượng mô hình con, kích thước của mỗi tập dữ liệu con, và cách kết hợp các dự đoán. Điều này có thể đòi hỏi nhiều thử nghiệm và tinh chỉnh để đạt được hiệu suất tốt nhất.
- Khó khăn trong việc giải thích (tăng độ phức tạp của mô hình): Do bagging cần kết hợp các dự đoán từ nhiều mô hình con, nó có thể tạo ra một mô hình tổng thể phức tạp hơn. Điều này có thể làm cho việc diễn giải và hiểu cấu trúc của mô hình trở nên khó khăn.

### Ứng dụng của Bagging:
Bagging có nhiều ứng dụng trong machine learning và các lĩnh vực liên quan. Dưới đây là một số ứng dụng phổ biến của bagging:

- Phân loại và Dự đoán: Bagging được sử dụng rộng rãi trong các bài toán phân loại và dự đoán, bao gồm phân loại hình ảnh, nhận diện ký tự, dự đoán giá cổ phiếu, dự đoán chuỗi thời gian, và nhiều bài toán khác.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20215641.png)
(Ứng dụng Bagging vào nhận diện kí tự)
- Random Forest: Random Forest là một phương pháp quan trọng dựa trên bagging, nó sử dụng một tập hợp của nhiều cây quyết định (decision trees) để thực hiện phân loại hoặc dự đoán. Random Forest thường được sử dụng trong các bài toán như phân loại ảnh, phân loại văn bản, và dự đoán sự cố trong hệ thống.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20211418.png)
(Ứng dụng của Bagging: Random Forest cho sự hiểu biết nhân quả)
- Học tập trên dữ liệu không cân bằng: Trong các tập dữ liệu không cân bằng, bagging có thể được sử dụng để tăng cường hiệu suất của các mô hình bằng cách tập trung vào việc huấn luyện trên các tập dữ liệu con có tỷ lệ cân bằng giữa các lớp.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20212627.png)
(Phân loại chuỗi thời gian mất cân bằng)
- Dự đoán thị trường tài chính: Trong lĩnh vực tài chính, bagging được sử dụng để dự đoán giá cổ phiếu, đánh giá rủi ro tín dụng, và các ứng dụng khác trong lĩnh vực dự báo và quản lý rủi ro.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20214316.png)
(Những lợi ích của phương pháp Bagging đối với mô hình dự đoán biến động thực tế)
- Xử lý dữ liệu y tế: Trong lĩnh vực y tế, bagging có thể được sử dụng để dự đoán nguy cơ bệnh lý, phân loại bệnh, dự đoán kết quả điều trị, và nhiều ứng dụng khác trong lĩnh vực dữ liệu y tế.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20215110.png)
(Lựa chọn đặc trưng ổn định bằng phương pháp Bagging trên dữ liệu y tế)
- Phát hiện gian lận: Trong lĩnh vực an ninh mạng và tài chính, bagging có thể được sử dụng để phát hiện gian lận, bao gồm phát hiện gian lận tín dụng, gian lận giao dịch, và các loại gian lận khác.
![Packaging status](https://github.com/TranAnh35/Data_mining/blob/dev/PNG/Screenshot%202024-03-02%20215315.png)
(Phát hiên gian lận thẻ tín dụng)