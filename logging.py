import logging
from datetime import datetime
import os
import pandas as pd
import atexit
import io
import boto3 
    
def write_logs(body, bucket, key):
    aws_id = "AKIAQJG6XJZ3IVIF46E7"
    aws_secret = "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"
    s3 = boto3.client('s3', aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret,region_name="ap-northeast-1")
    s3.put_object(Body=body, Bucket=bucket, Key=key)

            
def module_logging(log_dir):
    CURRENT_TIME_STAMP=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),

    LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"
    LOG_FILE_PATH=f"{log_dir}/{LOG_FILE_NAME}"
    #os.makedirs(log_dir,exist_ok=True)

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s')
    log_stringio = io.StringIO()
    file_handler=logging.StreamHandler(log_stringio)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    atexit.register(write_logs, body=log_stringio, bucket="data-science-valuematrix",key=f"{LOG_FILE_PATH}")
    return LOG_FILE_PATH,logger

def get_log_dataframe(file_path)->pd.DataFrame:
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))
    columns=["Time Stamp","Log Level","line number","file name","function name","message"]
    
    log_df=pd.DataFrame(data)

    log_df.columns=columns
    
    log_df["log_message"]=log_df['Time Stamp'].astype(str) +":$"+ log_df["message"] 

    return log_df[["log_message"]]
