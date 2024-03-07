# Báo Cáo Thực Hành

## Logistic Regression
### Giới thiệu
Logistic regression là một thuật toán được sử dụng để phân loại các quan sát theo các danh mục rời rạc. Thay vì có đầu ra là các giá trị liên tục như thuật toán Linear Regression, Logistic regression sử dụng hàm logistic sigmoid để trả về một giá trị biểu thị xác xuất có thể ánh xạ đến hai hay nhiều lớp rời rạc.

Logistic regression thường được phân thành 2 loại:
- Logistic Regression nhị phân( Binary Logistic Regression): được sử dụng để phân loại dữ liệu thành hai lớp, thường được gán nhãn là 0 và 1 (hoặc Negative và Positive). Ví dụ như dự đoán xem 1 email có phải là spam hay không hay dự đoán xem một học sinh sẽ đậu hay trượt kỳ thi.
- Multinomial Logistic Regression (Softmax Regression): Đây là loại Logistic Regression mở rộng, cho phép dự đoán dữ liệu vào ba hoặc nhiều hơn ba lớp không gian phân loại. Nó được sử dụng khi có ba lớp hoặc nhiều hơn trong biến phụ thuộc. Ví dụ phân loại email vào các hộp thư “rác”, “quan trọng”, “chính thức” …

### Logistic Regression
#### Activation function
Logistic Regression sử dụng hàm sigmoid làm hàm kích hoạt để chuyển đổi đầu ra của một hàm tuyến tính thành một giá trị xác suất nằm trong khoảng từ 0 đến 1.
Công thức toán học:

![!\[alt text\](image.png)](<PNG/Screenshot 2024-03-04 125441.png>)

Trong đó:
    S(z): Đầu ra trong khoảng từ 0 đến 1 ( giá trị xác suất ước lượng).
    z: đầu vào của hàm (là một hàm tuyến tính, ví dụ ax+b).
    e: hằng số Euler, là cơ số của logarit tự nhiên.

#### Hàm Sigmoid
Đồ thị hàm Sigmoid:

![alt text](<PNG/Screenshot 2024-03-04 125745.png>)

Trong bài toán Linear Regression, với 2 biến đầu vào dự báo là ![alt text](<PNG/Screenshot 2024-03-04 130200.png>) ta thu được hàm hồi quy ![alt text](<PNG/Screenshot 2024-03-04 130357.png>) và với ![alt text](<PNG/Screenshot 2024-03-04 130552.png>) là vector dòng của các hệ số hồi quy.

Chuyển tiếp qua hàm Sigmoid để dự báo xác suất và tạo tính phi tuyến cho mô hình hồi quy:
![alt text](<PNG/Screenshot 2024-03-04 131348.png>)

Với P(y = 1|x; w) là xác suất có điều kiện để xảy ra sự kiện y = 1 tương ứng với đầu vào x, w là trọng số.

#### Loss function
Xét bài toán phân lớp nhị phân (0:1)
Giả sử rằng xác suất để một điểm dữ liệu x rơi vào:
- class 1 là ![!\[alt text\](image.png)](<PNG/Screenshot 2024-03-04 131717.png>)
- class 0 là ![alt text](<PNG/Screenshot 2024-03-04 131725.png>)

Xác suất xảy ra tại điểm x_i theo hàm Sigmoid:

![alt text](<PNG/Screenshot 2024-03-04 132223.png>)

Theo công thức xác suất Bernoulli xác suất tổng quát cho mẫu cho cả hai trường hợp (0, 1) sẽ là:
![alt text](<PNG/Screenshot 2024-03-04 132538.png>)

Giả sử các quan sát trong bộ dữ liệu của chúng ta là độc lập. Khi đó xác suất đồng thời của toàn bộ các quan sát trong bộ dữ liệu sẽ bằng tích các xác suất tại từng điểm dữ liệu và bằng:

![alt text](<PNG/Screenshot 2024-03-04 132830.png>)

Biểu thức trên là hàm hợp lý (Likelihood Function) đo lường mức độ hợp lý (goodness of fit) của mô hình thống kê đối với dữ liệu. Điều ta muốn giá trị của hàm hợp lý phải lớn đồng nghĩa với các trường hợp tích cực phải có xác suất càng gần 1 và tiêu cực có xác suất gần bằng 0. Do đó mục tiêu của chúng ta là tìm w sao cho biểu thức (1) là lớn nhất. quá trình tìm nghiệm thực chất là giải bài toán tối ưu hàm hợp lý (Maximum Likelihood Function). Phương pháp tìm nghiệm w dựa trên hàm hợp lý còn được gọi là ước lượng hợp lý tối đa (Maximum Likelihood Estimation). 

Để tối ưu hàm (1) bớt khó khăn hơn ta sẽ logarith để chuyển tích sang tổng để tối ưu. Khi đó quy về bài toán tối ưu hàm Log Likelihood như sau:

![alt text](<PNG/Screenshot 2024-03-04 133022.png>)

Việc tìm giá trị cực đại của phương trình (1) tương ứng với bài toán tối ưu:

![alt text](<PNG/Screenshot 2024-03-04 133428.png>)

Vậy hàm Loss function có dạng:

![alt text](<PNG/Screenshot 2024-03-04 133553.png>)

Hàm mất mát trên còn được gọi là hàm Cross Entropy. Nó là một độ đo (metric) đo lường mức độ tương quan giữa phân phối xác suất dự báo (P(y_i=1), 1 – P(y_i=1)) và phân phối xác suất thực tế (y_i, 1-y_i) Giá trị của Cross Entropy sẽ càng nhỏ nếu hai phân phối xác suất càng sát nhau, tức là giá trị dự báo giống với thực tế nhất.

#### Gradient Descent
Để tìm ra nghiệm của Logistic regression thì chúng ta sẽ thực hiện cập nhật nghiệm trên từng điểm dữ liệu (x_i, y_i). Các điểm được lựa chọn một cách ngẫu nhiên ở mỗi lượt cập nhật. Phương pháp cập nhật gradient descent như vậy còn được gọi là Stochastic Gradient Descent.

![alt text](<PNG/Screenshot 2024-03-04 133926.png>)

Ta có công thức cập nhật cho Logistic Regression:

![alt text](<PNG/Screenshot 2024-03-04 134102.png>) với ![alt text](<PNG/Screenshot 2024-03-04 134531.png>)
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

### Kiểm tra thuật toán Bagging
Các bộ dữ liệu được chọn để so sánh được mô tả như sau:

```sh
STT     |   Data Name   | Sample| Feature| Class|
1       |DLBCl.csv      |77     |5469    |2     |
2       |Colon.csv      |62     |2000    |2     |
3       |Prostate.csv   |102    |6033    |2     |
```

Các bộ dữ liệu được chia làm 3 tập là Train - Validation - Test với tỉ lệ lần lượt 70 - 15 - 15 dựa trên phương pháp Stratified Sampling.

Khi tiến hành kiểm tra, chúng ta sẽ so sánh về độ chính xác của mô hình từ dố sẽ đưa ra các kết luận. Độ chính xác của mô hình sẽ được đánh giá dựa trên tập Validation. Dưới dây là kết quả của thuật toán với từng data.
- Với data: DLBCl.csv

![alt text](<PNG/Screenshot 2024-03-06 192149.png>)

- Với data: Colon.csv

![alt text](<PNG/Screenshot 2024-03-06 192219.png>)

- Với data: Prostate.csv

![alt text](<PNG/Screenshot 2024-03-06 192440.png>)

Do các data đều có rất nhiều feature nên không không thể thể hiện sự phụ thuộc của thuật toán vào các feature. Dưới đây là biểu đồ Boxplot thể hiện độ chính xác của thuật toán với từng data.
- Với data: DLBCl.csv

![!\[alt text\](<PNG/Screenshot 2024-03-06 192149.png>)](PNG/output_DLBCl.png)

- Với data: Colon.csv

![!\[alt text\](<PNG/Screenshot 2024-03-06 192219.png>)](PNG/output_Colon.png)

- Với data: Prostate.csv

![!\[alt text\](<PNG/Screenshot 2024-03-06 192440.png>)](PNG/output_Prostate.png)

Nhận xét:

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
