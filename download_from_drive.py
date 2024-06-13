from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def download_file_from_drive(file_id, destination):
    # Save credentials to a file
    credentials = os.getenv('GDRIVE_CREDENTIALS')
    with open('credentials.json', 'w') as f:
        f.write(credentials)

    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('credentials.json')
    drive = GoogleDrive(gauth)

    # Download the file
    downloaded_file = drive.CreateFile({'id': file_id})
    downloaded_file.GetContentFile(destination)

if __name__ == "__main__":
    file_id = 'your-google-drive-file-id'  # Replace with your file ID
    destination = 'validation.csv'  # Replace with your desired filename
    download_file_from_drive(file_id, destination)
