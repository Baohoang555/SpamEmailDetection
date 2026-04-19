import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Tải stopwords nếu chưa có
nltk.download('stopwords', quiet=True)

# 1. Định nghĩa lại các hàm tiền xử lý (phải giống hệt lúc train)
punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)
    return ' '.join(imp_words)

# 2. Tải Model và Tokenizer (dùng cache để không phải load lại nhiều lần)
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('spam_classifier_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# 3. Giao diện Web với Streamlit
st.title("📧 Hệ Thống Phân Loại Email Spam")
st.write("Nhập nội dung email vào bên dưới để kiểm tra xem đó là Thư rác (Spam) hay Thư hợp lệ (Ham).")

# Ô nhập liệu
user_input = st.text_area("Nội dung Email:", height=200)

if st.button("Kiểm tra Email"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung email!")
    else:
        with st.spinner('Đang phân tích...'):
            # Tiền xử lý văn bản
            clean_text = user_input.replace('Subject:', '')
            clean_text = remove_punctuations(clean_text)
            clean_text = remove_stopwords(clean_text)
            
            # Chuyển đổi thành sequence và padding
            max_len = 100 # Phải bằng max_len lúc train
            sequence = tokenizer.texts_to_sequences([clean_text])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
            
            # Dự đoán
            prediction_prob = model.predict(padded_sequence)[0][0]
            
            # Hiển thị kết quả
            if prediction_prob > 0.5:
                st.error(f"🚨 Cảnh báo: Đây là EMAIL SPAM (Độ tin cậy: {prediction_prob:.2%})")
            else:
                st.success(f"✅ An toàn: Đây là EMAIL BÌNH THƯỜNG (Độ tin cậy: {1 - prediction_prob:.2%})")