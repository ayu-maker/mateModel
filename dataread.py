import pandas as pd
import pymysql
from sqlalchemy import create_engine

# Create engine
engine = create_engine("mysql+pymysql://springstudent:springstudent@localhost:3306/login_system")

# Load dataset
query = "SELECT * FROM roommaterequest"
data = pd.read_sql(query, engine)

print(data.head())  # Make sure data is loading correctly
