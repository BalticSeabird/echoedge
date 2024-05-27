import os
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

def connect_with_engine():
    connection = create_engine("mysql+pymysql://{user}:{pw}@35.228.136.138/{db}".format(user = os.environ["DB_USER"] , pw=os.environ["DB_PASS"], db = os.environ["DB_NAME"]))
    return connection


def get_coords_from_table(session, start_time, end_time):
    query = text(f"SELECT * FROM svea WHERE time BETWEEN '{start_time}' AND '{end_time}';")
    result = session.execute(query)
    return result


def get_first_and_last_timestamp(session):
    
    first_query = text("SELECT time FROM svea ORDER BY time ASC LIMIT 1")
    first_result = session.execute(first_query)
    first_timestamp = first_result.fetchone()[0]

    
    last_query = text("SELECT time FROM svea ORDER BY time DESC LIMIT 1")
    last_result = session.execute(last_query)
    last_timestamp = last_result.fetchone()[0]    

    return first_timestamp, last_timestamp


if __name__ == "__main__":
    connection = connect_with_engine()
    Session = sessionmaker(bind=connection)
    session = Session()
    result = get_coords_from_table(session, "2024-05-21 00:00:00", "2024-05-21 00:10:00")

    rows = result.fetchall()
    for row in rows:
        print(row)
    
    # first_time, last_time = get_first_and_last_timestamp(session)
    # print(first_time, last_time)