import pyodbc
import os
import time
import numpy as np
import io
import pickle
#import random
#origdata for humanact12 #origname
#learning_data for cmu #filename
DB_SERVER = os.getenv("DB_SERVER", "semrob.database.windows.net")
DB_DATABASE = os.getenv("DB_DATABASE", "robotic-gifs")
DB_USERNAME = os.getenv("DB_USERNAME", "admin163")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")
DB_TRUST_SERVER_CERTIFICATE = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "Yes")
DB_CONNECTION_TIMEOUT = os.getenv("DB_CONNECTION_TIMEOUT", "30")

CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server=tcp:{DB_SERVER},{DB_PORT};"
    f"Database={DB_DATABASE};"
    f"Uid={DB_USERNAME};"
    f"Pwd={DB_PASSWORD or ''};"
    f"Encrypt={DB_ENCRYPT};"
    f"TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE};"
    f"Connection Timeout={DB_CONNECTION_TIMEOUT};"
)

def connect():
    if not DB_PASSWORD:
        raise RuntimeError("Missing DB_PASSWORD environment variable")

    max_attempts = 5
    delay = 1
    for attempt in range(max_attempts):
        try:
            conn = pyodbc.connect(CONNECTION_STRING)
            print(f"DB connected on attempt {attempt + 1}")
            return conn, conn.cursor()
        except pyodbc.Error as e:
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to DB after multiple attempts")

def init(connection, cur):
    global conn, cursor
    conn = connection
    cursor = cur

#submitusercheck
#get data (inference(labels included?) or learning)
datatables = ['cmu','jump_handsup','jump_vertical','run','sit','walk']
"""
userid = random.randint(10000, 99999)

test = ['jump_handsup','jump_vertical','run','sit','walk']

cmu = []
jump_handsup = []
jump_vertical = []
run = []
sit = []
walk = []
cursor.execute(f"SELECT * FROM dbo.cmu;")
for outside, filename, sequence_label, frame_labels_str, learning_data in cursor.fetchall():
    # Store each row as a list; keep learning_data as binary
    #table.append([outside, origname, sequence_label, frame_labels_str, learning_data]) #will get learning_data later
    cmu.append([outside, filename, sequence_label, frame_labels_str])
for x in test:
    cursor.execute(f"SELECT * FROM dbo.{x};")
    for origname in cursor.fetchall():
        if x == 'jump_handsup':
            jump_handsup.append([origname[0]])
        elif x == 'jump_vertical':
            jump_vertical.append([origname[0]])
        elif x == 'run':
            run.append([origname[0]])
        elif x == 'sit':
            sit.append([origname[0]])
        elif x == 'walk':
            walk.append([origname[0]])
"""
def change_tonpy(origname, dataset_id):
    filename = os.path.basename(origname)
    if dataset_id == 1:
        query = f"SELECT upperstatic FROM cmu WHERE filename = ?"
        cursor.execute(query, (filename,))
        row = cursor.fetchone()  # only fetch one row
        if row:
            return np.load(io.BytesIO(row[0]))  # load binary data into numpy array
    else:
        for table in datatables:
            if table == 'cmu':
                query = f"SELECT learning_data FROM {table} WHERE filename = ?"
            else:
                query = f"SELECT origdata FROM {table} WHERE origname = ?"

            cursor.execute(query, (filename,))
            row = cursor.fetchone()  # only fetch one row
            if row:
                return np.load(io.BytesIO(row[0]))  # load binary data into numpy array
                """
                with open(origname, 'wb') as f: #need to check path
                    f.write(np.load(io.BytesIO(row[0]))) #write binary data to file
                    print("npyfilegen")     
                """           

def get_binary(path):
    if path is not None and os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    else:
        return None

def add_history(user_input, model_input, motion_uploaded, model_output_file_name, table_number, task, userid, save_dict=None):
    # Build dynamic table name
    history_var = f'history{table_number}'

    history = pickle.dumps(save_dict) if save_dict is not None else None
    query = f"""
    INSERT INTO {history_var}
    (query, outputtext, outputmotion, userid, task, inputmotion, history)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    model_output_file_data = get_binary(model_output_file_name) #fix once hosted
    cursor.execute(query, (
        user_input, model_input, model_output_file_data, userid, task, motion_uploaded, history
    ))
    conn.commit()

def get_last_20_queries(comments, userid):
    for i in range(1, 8):
        history_var = f'history{i}'
        check_var = f'usercheck{i}'
        cursor.execute(f"""
            SELECT TOP 20 *
            FROM {history_var}
            WHERE userid = ?
            ORDER BY created_at DESC
        """, (userid,))
        last_20 = cursor.fetchall()
        for x in last_20:
            query = f"""
            INSERT INTO {check_var}
            (query, comments, outputtext, outputmotion, userid, task, inputmotion, created_at, history, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (x[0], comments, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]))
    conn.commit()

def score(score_value, setnum):
    history_var = f'history{setnum}'
    cursor.execute(f"""
        ;WITH latest AS (
            SELECT TOP (1) *
            FROM {history_var}
            ORDER BY created_at DESC
        )
        UPDATE latest
        SET outcome = ?
    """, (score_value,))
    conn.commit()
    

#def store_temp_data which will be mp4, npy
