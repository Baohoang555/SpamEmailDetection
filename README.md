# 📧 Hệ Thống Phân Loại Email Spam (LSTM)

Đây là một ứng dụng Web sử dụng Deep Learning để phân loại email là **Spam** (Thư rác) hoặc **Ham** (Thư bình thường). Dự án sử dụng mô hình mạng nơ-ron tuần hoàn **LSTM** (Long Short-Term Memory) để xử lý ngôn ngữ tự nhiên.

## 🚀 Tính năng
* **Tiền xử lý văn bản:** Tự động loại bỏ dấu câu, chuyển thành chữ thường và lọc bỏ stopwords.
* **Phân loại thời gian thực:** Nhập nội dung email và nhận kết quả dự đoán ngay lập tức với độ tin cậy (probability).
* **Giao diện trực quan:** Xây dựng bằng thư viện Streamlit, dễ dàng sử dụng và triển khai.

## 🛠 Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Tiền xử lý NLP:** NLTK, Scikit-learn
* **Web Framework:** Streamlit
* **Xử lý dữ liệu:** Pandas, Numpy

## 📋 Hướng dẫn cài đặt

1. **Clone project:**
   ```bash
   git clone [https://github.com/username/spam-classifier-project.git](https://github.com/username/spam-classifier-project.git)
   cd spam-classifier-project

2. **Kiến trúc mô hình:**
    🧠 Kiến trúc mô hình
```Mô hình được xây dựng với các lớp sau:

    Embedding Layer: Chuyển đổi từ ngữ thành vector không gian.

    LSTM Layer (16 units): Học ngữ cảnh và trình tự của các từ trong email.

    Dense Layer (32 units, ReLU): Lớp ẩn kết nối đầy đủ.

    Output Layer (Sigmoid): Trả về xác suất email là Spam.

Dự án được phát triển bởi Vũ Hoàng Bảo