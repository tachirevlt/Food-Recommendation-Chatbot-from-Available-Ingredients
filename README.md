# Food Recommendation Chatbot from Available Ingredients


## Tổng Quan
Pantry Genie là một chatbot hoạt động ngoại tuyến, gợi ý món ăn dựa trên nguyên liệu mà người dùng nhập vào. Dự án bao gồm hai thành phần chính: một script huấn luyện và tiền xử lý dữ liệu, và một script chatbot để tương tác với người dùng. Chatbot không cần kết nối internet, có thể sử dụng offline.

## Chức Năng
- Gợi ý món ăn dựa trên nguyên liệu người dùng cung cấp (hoạt động ngoại tuyến, không cần mạng).
- Hỗ trợ cả phương pháp dự đoán supervised (dựa trên mô hình) và unsupervised (dựa trên độ tương đồng).
- Hiển thị chi tiết món ăn (tên, nguyên liệu, hướng dẫn, link, nguồn) khi người dùng chọn.

## Quy Trình Chính

### Quy Trình Huấn Luyện & Tiền Xử Lý
1. **Load Dataset**:
   - Load datasets từ HuggingFace để lấy bộ dữ liệu mẫu về món ăn (`Schmitz005/recipe_nlg_dataset_sample`).
2. **Tiền Xử Lý**:
   - Chuyển dữ liệu thành DataFrame và lưu ra file JSON.
   - Trích xuất nguyên liệu thô (`raw_ingredients`) từ trường `NER`, loại bỏ số lượng và từ dư thừa.
   - Tạo trường `raw_str` (chuỗi nguyên liệu thô cho mỗi món ăn).
3. **Vector Hóa**:
   - Sử dụng TF-IDF (hoặc chuyển sang CountVectorizer nếu TF-IDF lỗi) để biến đổi `raw_str` thành ma trận đặc trưng.
4. **Huấn Luyện Mô Hình**:
   - Chia dữ liệu thành tập train và test.
   - Huấn luyện mô hình `RandomForestClassifier` để dự đoán tên món ăn từ nguyên liệu.
   - Đánh giá và in ra độ chính xác (accuracy).
5. **Lưu Lại**:
   - Lưu vectorizer, ma trận đặc trưng, mô hình đã huấn luyện, và DataFrame đã xử lý ra các file để `chatbot.py` sử dụng.

### Quy Trình Chatbot
1. **Tải Mô Hình & Dữ Liệu**:
   - Đọc các file đã huấn luyện trước: `recipe_model.pkl`, `vectorizer.pkl`, `features_matrix.pkl`, và `recipes_df.pkl`.
2. **Chuẩn Hóa Nguyên Liệu**:
   - Ánh xạ nguyên liệu tiếng Việt sang tiếng Anh cơ bản (ví dụ: "gà" → "chicken").
3. **Gợi Ý Món Ăn**:
   - **Supervised**: Dự đoán tên món ăn phù hợp nhất dựa trên mô hình đã huấn luyện.
   - **Unsupervised**: Tính toán độ tương đồng cosine giữa nguyên liệu người dùng và các món ăn, lấy top K món có điểm cao nhất.
4. **Hiển Thị Chi Tiết Món Ăn**:
   - Khi người dùng chọn số thứ tự, hiển thị tên món, nguyên liệu, hướng dẫn nấu, link (nếu có), và nguồn.
5. **Vòng Lặp**:
   - Nhận nguyên liệu từ người dùng, gợi ý món, cho phép chọn, và lặp lại cho đến khi nhập 'quit' để thoát.

## Hướng Dẫn Cài Đặt Và Chạy Bot

### Yêu Cầu Hệ Thống
- Python 3.10 hoặc cao hơn (khuyến nghị 3.12.3).
- Môi trường Windows, Linux hoặc macOS với terminal/cmd.
- Git đã cài đặt (tải từ [git-scm.com](https://git-scm.com/) nếu chưa có).

### Cài Đặt
1. **Clone Repository Từ GitHub**:
   - Mở terminal, chạy lệnh sau để lấy toàn bộ code:
     ```
     git clone https://github.com/tachirevlt/Food-Recommendation-Chatbot-from-Available-Ingredients.git
     ```
   - Chuyển vào thư mục vừa clone:
     ```
     cd example
     ```
2. **Cài Python**:
   - Nếu chưa có, tải từ [python.org](https://www.python.org/downloads/) và cài đặt (chọn thêm Python vào PATH).
3. **Cài Thư Viện**:
   - Trong terminal, chạy:
     ```
     pip install pandas numpy scikit-learn joblib
     ```
### Chạy Bot Cục Bộ
1. **Huấn Luyện Và Tiền Xử Lý**:
   - Chạy script huấn luyện:
     ```
     python train_chatbot.py
     ```
2. **Chạy Chatbot**:
   - Chạy script chatbot:
     ```
     python chatbot.py
     ```
   - Nhập nguyên liệu (ví dụ: "gà, cà rốt") hoặc gõ 'quit' để thoát.
   - Chọn số thứ tự để xem chi tiết món ăn.

## Lưu Ý
- Chatbot hoạt động ngoại tuyến phụ thuộc vào dataset dùng để huấn luyện, cần file `.pkl` đã huấn luyện trước.
- Độ chính xác phụ thuộc vào chất lượng dataset và tiền xử lý.
- Điều chỉnh dictionary `ing_map` trong `chatbot.py` để thêm bản dịch tiếng Việt - tiếng Anh nếu cần.

## Cải Tiến Trong Tương Lai
- Hỗ trợ thêm bộ lọc (ví dụ: thời gian nấu ăn).
- Mở rộng bản đồ nguyên liệu với dictionary lớn hơn.
- Tích hợp telegram hoặc ứng dụng cục bộ.
- Bổ sung tiếng việt cho chatbot

## Author

[![GitHub](https://img.shields.io/badge/GitHub-tachirevlt-blue?logo=github)](https://github.com/tachirevlt)

