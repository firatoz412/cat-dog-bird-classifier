import mysql.connector
from mysql.connector import Error
import os

def getDatabase():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        
        if connection.is_connected():
            return connection
    except Error as e:
        print("Veri Tabanı Bağlantı Hatası:", e)
        return None