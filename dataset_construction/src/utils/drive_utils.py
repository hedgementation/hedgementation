# Google Drive API setup
import os
import pickle
import re
import time
from typing import Set
from urllib.request import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_httplib2 import AuthorizedHttp
import google_auth_httplib2
import httplib2


from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
import pickle
import os


SCOPES = ['https://www.googleapis.com/auth/drive']


_creds = None
def get_creds():
    global _creds
    if _creds and _creds.valid:
        return _creds
    creds = None
    # Token file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=8085)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    _creds = creds
    return creds

def get_http(creds=None, http=None):
    if creds is None:
        creds = get_creds()
    return AuthorizedHttp(credentials=creds, http=httplib2.Http())

def get_drive_service(creds=None, http=None):
    if creds is None and http is None:
        creds = get_creds()
    if creds:
        return build('drive', 'v3', credentials=creds)
    if http:
        return build('drive', 'v3', http=http)

def get_or_create_folder(service, folder_name, parent_id=None):
    """Get folder ID or create if it doesn't exist."""
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    items = results.get('files', [])
    
    if items:
        return items[0]['id']
    
    # Create folder if it doesn't exist
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    folder = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()
    
    return folder.get('id')


def get_folder_id_from_path(service, folder_path):
    """Convert folder path like 'Hedgementation_Dataset_1.0\\X' to folder ID."""
    # Split path by both forward and backslash
    parts = folder_path.replace('\\', '/').split('/')
    
    parent_id = None
    for part in parts:
        if part:  # Skip empty parts
            parent_id = get_or_create_folder(service, part, parent_id)
    
    return parent_id


def get_drive_folder_name(folder_id, creds=None):
    """Return the display name of a Drive folder given its ID (read-only)."""
    service = get_drive_service(creds=creds)
    metadata = service.files().get(
        fileId=folder_id,
        fields='name',
        supportsAllDrives=True,
    ).execute()
    return metadata.get('name')


def list_drive_subfolders(root_id, creds=None):
    """List immediate subfolders of a Drive folder (read-only).

    Returns a list of (id, name) tuples. Supports Shared/Team Drives.
    """
    service = get_drive_service(creds=creds)
    query = (
        f"'{root_id}' in parents and "
        "mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    subfolders = []
    page_token = None
    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        for f in response.get('files', []):
            subfolders.append((f['id'], f['name']))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return subfolders


def upload_file_to_drive(service, file_path, folder_path, logger, new_filename=None):
    """Upload a file to a specific Google Drive folder."""
    try:
        # Get folder ID
        folder_id = get_folder_id_from_path(service, folder_path)
        
        # Set filename
        filename = new_filename if new_filename else os.path.basename(file_path)
        
        # Check if file already exists in folder
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields='files(id)').execute()
        existing_files = results.get('files', [])
        
        # Delete existing file if it exists
        for existing_file in existing_files:
            service.files().delete(fileId=existing_file['id']).execute()
            logger.info(f"Deleted existing {filename} from {folder_path}")
        
        # Upload new file
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(file_path, mimetype='application/geo+json')
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        logger.info(f"Uploaded {filename} to {folder_path} (ID: {file.get('id')})")
        return file.get('id')
        
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to {folder_path}: {e}")
        raise

def download_file_from_drive(service, file_path, folder_path, logger, new_filename=None):
    """Upload a file to a specific Google Drive folder."""
    try:
        # Get folder ID
        folder_id = get_folder_id_from_path(service, folder_path)
        
        # Set filename
        filename = new_filename if new_filename else os.path.basename(file_path)
        
        # Check if file already exists in folder
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields='files(id)').execute()
        existing_files = results.get('files', [])
        
        # Delete existing file if it exists
        for existing_file in existing_files:
            service.files().delete(fileId=existing_file['id']).execute()
            logger.info(f"Deleted existing {filename} from {folder_path}")
        
        # Upload new file
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(file_path, mimetype='application/geo+json')
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        logger.info(f"Uploaded {filename} to {folder_path} (ID: {file.get('id')})")
        return file.get('id')
        
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to {folder_path}: {e}")
        raise

async def save_and_upload_metadata(gdf, satellite_folder, hedgerow_folder, logger):
    """Save GeoDataFrame as metadata.geojson and upload to both Drive folders."""
    try:
        # Save to temporary local file
        temp_path = "temp_metadata.geojson"
        gdf.to_file(temp_path, driver='GeoJSON')
        logger.info(f"Saved metadata GeoJSON locally to {temp_path}")
        
        # Get Drive service
        service = get_drive_service()
        
        # Upload to both folders
        upload_file_to_drive(service, temp_path, satellite_folder, logger, "metadata.geojson")
        upload_file_to_drive(service, temp_path, hedgerow_folder, logger, "metadata.geojson")
        
        # Clean up temp file
        os.remove(temp_path)
        logger.info("Cleaned up temporary metadata file")
        
    except Exception as e:
        logger.error(f"Failed to save and upload metadata: {e}")
        raise

def get_existing_ids_from_drive(folder_id: str, suffix_indicator: str, logger) -> Set[str]:
    """
    Fetches file names from a Google Drive folder and extracts feature IDs.
    Includes parameters to support Shared Drives and debugging to see what files are found.
    """
    if not folder_id:
        return set()

    logger.info(f"Checking existing files in Drive Folder ID: {folder_id}...")
    
    try:
        # Request specific scopes to ensure we can see files created by others (like Earth Engine)
        service = get_drive_service()
        
        existing_ids = set()
        page_token = None
        
        # 'q' parameter: Search within the folder and ensure not trashed
        query = f"'{folder_id}' in parents and trashed = false"

        while True:
            # We must add supportAllDrives and includeItemsFromAllDrives 
            # to see files if this folder is on a Shared/Team Drive.
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token,
                supportsAllDrives=True,         # Crucial for Shared Drives
                includeItemsFromAllDrives=True  # Crucial for Shared Drives
            ).execute()
            
            files_found = response.get('files', [])
            
            # DEBUG: Print the first few files found to verify visibility
            if page_token is None and files_found:
                 logger.info(f"DEBUG: First 5 files seen by API in this folder: {[f['name'] for f in files_found[:5]]}")

            for file in files_found:
                name = file.get('name')
                
                # Check for standard export format: "{id}_X.tif" or "{id}_Y.tif"
                if name.endswith('.tif') and suffix_indicator in name:
                    base_name = name.rsplit('.', 1)[0] # remove extension
                    
                    # Ensure the suffix is actually at the end of the ID part
                    if base_name.endswith(suffix_indicator):
                        extracted_id = base_name.rsplit(suffix_indicator, 1)[0]
                        existing_ids.add(extracted_id)

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
                
        logger.info(f"Found {len(existing_ids)} matching '{suffix_indicator}' files in folder {folder_id}")
        return existing_ids


    
    except Exception as e:
        logger.error(f"Error fetching files from Drive: {e}")
        return set()
    
def verify_files(folder_id, max_id):
    service = get_drive_service() 
    print(f"\n--- Processing Folder: {folder_id} ---")

    query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"


    page_token = None

    seen_ids = set()
    while True:
        try:
            query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
            
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token
            ).execute()
            
            items = results.get("files", [])

            if not items:
                print("No more files found in this folder.")
                break
            
            file_names = [file.get("name") for file in items]
            file_ids = [int(file_name.split("_")[0]) for file_name in file_names if file_name[-4:] == ".tif"]
            seen_ids.update(file_ids)
                

            page_token = results.get("nextPageToken")
            if not page_token:
                break
        except Exception as error:
            print(f"An error occurred: {error}")
            break

    expected_ids = set([i for i in range(0, max_id)])

    missing_ids = expected_ids.difference(seen_ids)

    if len(missing_ids) > 0:
        print("The following IDs were missing:")
        print(missing_ids)
    else:
        print("All expected IDs were found!")
            
        


def move_files(source_ids, destination_id, copy_only=True):

    service = get_drive_service()    
    for source_id in source_ids:
        print(f"\n--- Processing Source Folder: {source_id} ---")
        
        page_token = None
        while True:
            try:
                # Query to list files in the specific source folder
                # 'trashed = false' ensures we don't try to move deleted items
                # 'mimeType != "application/vnd.google-apps.folder"' ensures we only move files, not subfolders
                query = f"'{source_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
                
                results = service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, parents)",
                    pageToken=page_token
                ).execute()
                
                items = results.get("files", [])

                if not items:
                    print("No more files found in this folder.")
                    break

                for file in items:
                    file_id = file.get("id")
                    file_name = file.get("name")
                    
                    if copy_only:
                        # --- COPY MODE ---
                        # Create a copy metadata with the new parent
                        file_metadata = {
                            'name': file_name,
                            'parents': [destination_id]
                        }
                        service.files().copy(
                            fileId=file_id,
                            body=file_metadata
                        ).execute()
                        print(f"Copied: {file_name}")
                        
                    else:
                        # --- MOVE MODE ---
                        previous_parents = ",".join(file.get("parents"))
                        service.files().update(
                            fileId=file_id,
                            addParents=destination_id,
                            removeParents=previous_parents,
                            fields="id, parents"
                        ).execute()
                        print(f"Moved: {file_name}")

                    # Small sleep to prevent hitting API rate limits
                    time.sleep(0.1)

                page_token = results.get("nextPageToken")
                if not page_token:
                    break
                    
            except Exception as error:
                print(f"An error occurred: {error}")
                break