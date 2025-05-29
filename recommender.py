# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# class RecommenderSystem:
#     def __init__(self, csv_path=None, dataframe=None):
#         if dataframe is not None:
#             self.df = dataframe
#         elif csv_path:
#             self.df = pd.read_csv(csv_path)
#         else:
#             raise ValueError("You must provide either a CSV path or a DataFrame.")
#
#         self.df.fillna("", inplace=True)
#
#         self.df['combined_features'] = (
#             self.df['occupation'].astype(str) + " " +
#             self.df['sleep_schedule'].astype(str) + " " +
#             self.df['personality'].astype(str) + " " +
#             self.df['cleanliness'].astype(str) + " " +
#             self.df['budget_range'].astype(str) + " " +
#             self.df['accommodation_type'].astype(str) + " " +
#             self.df['preferred_area'].astype(str) + " " +
#             self.df['interests'].astype(str) + " " +
#             self.df['social_activity_level'].astype(str)
#         )
#
#         self.vectorizer = CountVectorizer()
#         self.feature_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
#         self.similarity_matrix = cosine_similarity(self.feature_matrix)
#
#     def get_recommendations(self, person_index, top_n=5):
#         similarity_scores = list(enumerate(self.similarity_matrix[person_index]))
#         sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#         sorted_scores = [score for score in sorted_scores if score[0] != person_index]
#         top_indices = [i[0] for i in sorted_scores[:top_n]]
#         return self.df.iloc[top_indices][['name', 'occupation', 'interests', 'preferred_area','budget_range']]
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSystem:
    def __init__(self, csv_path=None, dataframe=None):
        if dataframe is not None:
            self.df = dataframe
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("You must provide either a CSV path or a DataFrame.")

        self.df.fillna("", inplace=True)


        # Combine relevant features into a single string for each user
        self.df['combined_features'] = (
            self.df['occupation'].astype(str) + " " +
            self.df['sleepSchedule'].astype(str) + " " +
            self.df['personality'].astype(str) + " " +
            self.df['cleanliness'].astype(str) + " " +
            self.df['budgetRange'].astype(str) + " " +
            self.df['accommodationType'].astype(str) + " " +
            self.df['preferredArea'].astype(str) + " " +
            self.df['interests'].astype(str) + " " +

            self.df['socialActivityLevel'].astype(str)
        )

        # Convert text into feature vectors
        self.vectorizer = CountVectorizer()
        self.feature_matrix = self.vectorizer.fit_transform(self.df['combined_features'])

        # Compute cosine similarity between all entries
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    # def get_recommendations(self, person_index, top_n=5):
    #     # Get similarity scores for the person
    #     similarity_scores = list(enumerate(self.similarity_matrix[person_index]))
    #
    #     # Sort by similarity (excluding the person themself)
    #     sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    #     sorted_scores = [score for score in sorted_scores if score[0] != person_index]
    #
    #     # Get top N recommendations
    #     top_indices = [i[0] for i in sorted_scores[:top_n]]
    #
    #     return self.df.iloc[top_indices][['name', 'occupation', 'interests', 'preferred_area','gender']]

    def get_recommendations_from_input(self, user_input, top_n=5):
        # Step 1: Filter the DataFrame based on strict fields
        strict_fields = ['accommodationType', 'preferredArea', 'gender', 'personality']
        filtered_df = self.df.copy()

        for field in strict_fields:
            if field in user_input:
                filtered_df = filtered_df[filtered_df[field] == user_input[field]]

        # If no match found, return empty
        if filtered_df.empty:
            return pd.DataFrame(columns=['name', 'occupation', 'interests', 'preferredArea', 'gender'])

        # Step 2: Create combined input features from the input user
        input_combined = (
                str(user_input.get('occupation', '')) + " " +
                str(user_input.get('sleepSchedule', '')) + " " +
                str(user_input.get('personality', '')) + " " +
                str(user_input.get('cleanliness', '')) + " " +
                str(user_input.get('budgetRange', '')) + " " +
                str(user_input.get('accommodationType', '')) + " " +

                str(user_input.get('preferredArea', '')) + " " +

                str(user_input.get('interests', '')) + " " +
                str(user_input.get('socialActivityLevel', ''))
        )

        # Step 3: Re-vectorize only the filtered data
        filtered_combined = filtered_df['combined_features']
        filtered_vectors = self.vectorizer.transform(filtered_combined)

        # Step 4: Transform input and compute cosine similarity
        input_vector = self.vectorizer.transform([input_combined])
        similarity_scores = cosine_similarity(input_vector, filtered_vectors)[0]

        # Step 5: Get top-N similar users in filtered set
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        return filtered_df.iloc[top_indices][['name', 'occupation','personality' ,'cleanliness','interests','accommodationType', 'preferredArea', 'gender','budgetRange','sleepSchedule','socialActivityLevel','photoRoommate','id' ,'age']]


# # Example usage (optional testing)
# if __name__ == "__main__":
#     from sqlalchemy import create_engine
#
#     # DB configuration
#     db_user = "springstudent"
#     db_password = "springstudent"
#     db_host = "localhost"
#     db_name = "login_system"
#
#     # Create engine and load data
#     engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
#     query = "SELECT * FROM roommaterequest"
#     df = pd.read_sql(query, con=engine)
#
#     print("Columns in your DataFrame:", df.columns.tolist())  # Check column names
#
#     # Initialize model with MySQL data
#     model = RecommenderSystem(dataframe=df)
#
#     # Show recommendations for first person (index 0)
#     recommendations = model.get_recommendations(person_index=0, top_n=3)
#     print(recommendations)
