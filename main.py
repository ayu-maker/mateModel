from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from flask_cors import CORS
import pandas as pd
from recommender import RecommenderSystem

app = Flask(__name__)
CORS(app)

# DB config
db_user = "root"
db_password = "KGAuDpHeslKEvFGdDHykCKezDkPRaMwZ"
db_host = "caboose.proxy.rlwy.net:55112"
db_name = "railway"

# SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=utf8mb4")

def load_data_from_mysql():
    query = "SELECT * FROM roommaterequest"
    df = pd.read_sql(query, con=engine)
    return df

@app.route("/recommend", methods=["POST"])
def recommend():
    user_data = request.get_json()
    if not user_data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        df = load_data_from_mysql()
        model = RecommenderSystem(dataframe=df)
        top_n = int(user_data.get("top_n", 5))
        recommendations = model.get_recommendations_from_input(user_input=user_data, top_n=top_n)
        recommendations = recommendations.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
        return jsonify({"matches": recommendations.to_dict(orient="records")})
    except Exception as e:
        app.logger.error(f"Recommendation error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
