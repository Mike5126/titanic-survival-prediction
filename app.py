import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load mô hình và scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy 8 giá trị từ form trên web
    features = [float(x) for x in request.form.values()]
    
    # Biến thành mảng numpy và đưa qua scaler chuẩn hóa
    final_features = [np.array(features)]
    final_features_scaled = scaler.transform(final_features)
    
    # Dự đoán
    prediction = model.predict(final_features_scaled)[0]
    
    # Dịch kết quả
    if prediction == 1:
        result_text = "Dự đoán: Hành khách này có khả năng SỐNG SÓT 🚢"
    else:
        result_text = "Dự đoán: Hành khách này KHÔNG THỂ qua khỏi 💔"

    return render_template('home.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)