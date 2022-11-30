# import os
# import io
# from Google import Create_Service
# from googleapiclient.http import MediaIOBaseDownload



# # import the required libraries
# from __future__ import print_function
# import pickle
# import os.path
# import io
# import shutil
# import requests
# from mimetypes import MimeTypes
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly'] #['https://wwww.googleapis.com/auth/drive']


# file_ids = ['1-Y2zE1sa6lkDG5c3IZMMioo-TR0rc7M8']
# file_names = ['model.h5']
# # service = Create_Service(CLIENT_SECRET_FILE, API_NAME, SCOPES)


# # for file_id, file_name in zip(file_ids, file_names):
# #     request = service.files().get_media(fileId=file_id)

# #     fh = io.BytesIO()
# #     downloader = MediaIOBaseDownload(fd=fh, request=request)

# #     done = False

# #     while not done:
# #         status, done = downloader.next_chunk()
# #         print('Download progress {0}'.format(status.progress()*100))

# #     fh.seek(0)

# #     with open(os.path.join('.', file_name), 'wb') as f:
# #         f.write(fh.read())
# #         f.close()






# # Define the scopes

# flow = InstalledAppFlow.from_client_secrets_file(
#             'client_secret_googledemo.json', SCOPES)
# creds = flow.run_local_server(port=0)


# # Connect to the API service
# service = build(API_NAME, API_VERSION, credentials=creds)

# for file_id, file_name in zip(file_ids, file_names):
#     request = service.files().get_media(fileId=file_id)

#     fh = io.BytesIO()
#     downloader = MediaIOBaseDownload(fd=fh, request=request)

#     done = False

#     while not done:
#         status, done = downloader.next_chunk()
#         print('Download progress {0}'.format(status.progress()*100))

#     fh.seek(0)

#     with open(os.path.join('.', file_name), 'wb') as f:
#         f.write(fh.read())
#         f.close()


from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

CLIENT_SECRET_FILE = 'client_secret_googledemo.json'
API_NAME = 'AlzheimersPrediction'
API_VERSION = 'v3'
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret_googledemo.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build(API_NAME, API_VERSION, credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()
