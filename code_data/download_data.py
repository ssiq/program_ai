import paramiko
from code_data.constants import spider_key_path, remote_db_path, spider_ip, spider_port, local_db_path
import os
import logging

def get_remote_data():

    if not os.path.exists(spider_key_path):
        logging.error("Private key not exists.")
        return

    key = paramiko.RSAKey.from_private_key_file(spider_key_path)
    t = paramiko.Transport((spider_ip, spider_port))
    t.connect(username= 'lf', pkey= key)
    sftp = paramiko.SFTPClient.from_transport(t)
    localpath = local_db_path
    logging.info("Start download remote scrapyOJ.db file.")
    sftp.get(remote_db_path, localpath)
    logging.info("End download remote scrapyOJ.db file.")
    t.close()
