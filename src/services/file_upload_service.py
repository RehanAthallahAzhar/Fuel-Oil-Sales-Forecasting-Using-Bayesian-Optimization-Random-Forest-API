import pandas as pd
import os

from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from src.utils.http_status_code import HTTPStatusCode
from src.response import JSON
from src.config.config import Config

class Upload:
    def excel():
        try:
            testFile = Config().test_file
            folder_path = testFile.FOLDER_PATH[0]
            saved_filename = testFile.SAVED_FILENAME[0]

            if 'file' not in request.files:
                return JSON(HTTPStatusCode.BAD_REQUEST, "No file part")
            
            file = request.files['file']
            if file.filename == '':
                return JSON(HTTPStatusCode.BAD_REQUEST, "No selected file")
            
            if file:
                file_path = os.path.join(folder_path, f"{saved_filename}.xlsx")
                file.save(file_path)
                
                # Convert Excel to CSV
                df = pd.read_excel(file_path) 
                csv_filename = f"{os.path.splitext(saved_filename)[0]}.csv"
                csv_path = os.path.join(folder_path, csv_filename)
                df.to_csv(csv_path, index=False)  # Save as CSV
                
                # Delete the uploaded Excel file
                os.remove(file_path)
                
                return JSON(HTTPStatusCode.OK, "Upload sucessfully")
        
        except ValueError as ve:
            return JSON(HTTPStatusCode.BAD_REQUEST, f"Upload failed: {str(ve)}")

        except Exception as e:
            return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}") 

