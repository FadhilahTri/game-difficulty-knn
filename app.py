from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')
le_dict = joblib.load('model/label_encoders.pkl')

# Mapping hasil prediksi
engagement_map = {0: 'High', 1: 'Low', 2: 'Medium'}
engagement_desc = {
    'High': 'Pemain sangat aktif dan terlibat dalam game!',
    'Low': 'Pemain jarang aktif dan kurang terlibat dalam game.',
    'Medium': 'Pemain cukup aktif dan terlibat dalam game.'
}
engagement_color = {
    'High': 'success',
    'Low': 'danger',
    'Medium': 'warning'
}
engagement_icon = {
    'High': '🏆',
    'Low': '😴',
    'Medium': '🎮'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        age = float(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        game_genre = request.form['game_genre']
        playtime = float(request.form['playtime'])
        in_game_purchases = int(request.form['in_game_purchases'])
        game_difficulty = request.form['game_difficulty']
        sessions_per_week = float(request.form['sessions_per_week'])
        avg_session = float(request.form['avg_session'])
        player_level = float(request.form['player_level'])
        achievements = float(request.form['achievements'])

        # Encode kategorikal
        gender_enc = le_dict['Gender'].transform([gender])[0]
        location_enc = le_dict['Location'].transform([location])[0]
        genre_enc = le_dict['GameGenre'].transform([game_genre])[0]
        difficulty_enc = le_dict['GameDifficulty'].transform([game_difficulty])[0]

        # Susun fitur
        features = np.array([[age, gender_enc, location_enc, genre_enc,
                               playtime, in_game_purchases, sessions_per_week,
                               avg_session, player_level, achievements,
                               difficulty_enc]])

        # Normalisasi
        features_scaled = scaler.transform(features)

        # Prediksi
        pred = model.predict(features_scaled)[0]
        result = engagement_map[pred]

        return render_template('result.html',
                               result=result,
                               description=engagement_desc[result],
                               color=engagement_color[result],
                               icon=engagement_icon[result],
                               input_data=request.form)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)