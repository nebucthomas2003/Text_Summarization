from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json

def authenticate_drive():
    # Save credentials to a file
    credentials = os.getenv('GDRIVE_CREDENTIALS')
    if not credentials:
        raise ValueError("GDRIVE_CREDENTIALS environment variable not set.")
    
    with open('credentials.json', 'w') as f:
        f.write(credentials)
    
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('credentials.json')
    drive = GoogleDrive(gauth)
    return drive

def list_files_in_folder(drive, folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list

def download_files_from_folder(drive, folder_id, destination_folder):
    file_list = list_files_in_folder(drive, folder_id)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for file in file_list:
        file_path = os.path.join(destination_folder, file['title'])
        file.GetContentFile(file_path)
        print(f"Downloaded {file['title']} to {file_path}")

if __name__ == "__main__":
    folder_id = '1FV4QtW0oX-mXCCdgsRxEASrETJG-5N4_'  # Replace with your folder ID
    destination_folder = 'env_files'  # The destination folder to save the files
    
    drive = authenticate_drive()
    download_files_from_folder(drive, folder_id, destination_folder)
    print("All files downloaded.")
