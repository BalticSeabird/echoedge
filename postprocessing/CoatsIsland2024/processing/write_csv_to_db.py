import pandas as pd
import sqlite3
import os
from pathlib import Path

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn



def insert_to_db(input, output, tablename): 
    input = input.reset_index()
    con_local = create_connection(output)
    input.to_sql(tablename, con_local, if_exists='append')



# Delete old version if existing
if os.path.exists("../../out/CoatsData2024.db"):
    os.remove("../../out/CoatsData2024.db")


# Create empty db
con_local = create_connection("../../out/CoatsData2024.db")

# Read all csv files in the folder
files = Path("../../../../../../../mnt/BSP_NAS2_work/Acoustics_output_data/Echopype_results/Hudson2024/run9dec24/csv/").glob("*.csv")

# Loop through all files
for file in files:
    print(file)
    df = pd.read_csv(file)
    insert_to_db(df, "../../out/CoatsData2024.db", "data")