import argparse
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

def main(folder_id, destination_folder):
    """
    Downloads metadata.geojson from a Google Drive folder to a local folder.
    
    Args:
        folder_id: Google Drive folder ID
        local_folder: Local destination folder path
    """
    # Authenticate and create drive instance
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver for authentication
    drive = GoogleDrive(gauth)
    
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and title='metadata.geojson' and trashed=false"
    }).GetList()
    
    if not file_list:
        print("metadata.geojson not found in the specified folder")
        return
    
    file = file_list[0]
    
    # Download the file
    output_path = os.path.join(destination_folder, 'metadata.geojson')
    file.GetContentFile(output_path)
    
    print(f"Downloaded metadata.geojson to {output_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("destination_folder", help="The ultimate folder to save the downloaded information to")
  parser.add_argument("--folder_id", default="1zZZN9gCRqUApYsXNsM4UsDZDLbEO7waK", help="The id of the google drive folder whose contents are to be downloaded")

  
  args = parser.parse_args()

  
  main(
    folder_id = args.folder_id,
    destination_folder = args.destination_folder
  )
  