import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred')
def pred():
  return render_template('prediksi.html')


@app.route('/about')
def about():
  return render_template('about.html')
  


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil & konversi input ke float
        price_open = float(request.form.get('price_open', '').strip())
        price_high = float(request.form.get('price_high', '').strip())
        price_low  = float(request.form.get('price_low',  '').strip())
    except (ValueError, AttributeError):
        return render_template('prediksi.html',
                               prediction_text="Input tidak valid. Pastikan semua kolom berisi angka.")


    # Validasi logika harga
    if price_high < price_open:
        output = "Price High tidak boleh lebih kecil dari Price Open."
    elif price_low > price_high:
        output = "Price Low tidak boleh lebih besar dari Price High."
    else:
        # Siapkan fitur dalam bentuk 2D array untuk sklearn: [[open, high, low]]
        final_features = np.array([[price_open, price_high, price_low]])
        prediction = model.predict(final_features)
        output = f"Prediksi Harga Bitcoin adalah: $ {round(float(prediction[0]), 2)}"

    return render_template('prediksi.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)