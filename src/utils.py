import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException

load_dotenv()

def read_sql_data():
    try:
        logging.info("Connecting to MySQL database")

        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            db=os.getenv("MYSQL_DATABASE"),
            port=int(os.getenv("MYSQL_PORT"))
        )

        logging.info("Connection successful")

        query = "SELECT * FROM loan_data"   # change if needed
        df = pd.read_sql_query(query, connection)

        logging.info(f"Data fetched successfully: {df.shape}")

        return df

    except Exception as e:
        raise CustomException(e, sys)
    
print(read_sql_data())