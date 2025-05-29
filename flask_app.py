# flask_app.py
from flask import Flask, request, jsonify
import pandas as pd
from recommender import RecommenderSystem  # Assuming your model file

app = Flask(__name__)
model = RecommenderSystem(csv_path="matedata.csv")  # Make sure filename is correct!


@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json
    recommendations = model.get_recommendations(user_input)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)