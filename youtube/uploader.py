import logging
import os
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_youtube_client(client_secrets: str, token_file: str) -> Optional[any]:
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as token:
            token.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: List[str],
    client_secrets: str,
    token_file: str,
) -> Optional[str]:
    if not os.path.exists(video_path):
        logging.error("Video file not found: %s", video_path)
        return None

    youtube = get_youtube_client(client_secrets, token_file)
    body = {
        "snippet": {"title": title, "description": description, "tags": tags, "categoryId": "28"},
        "status": {"privacyStatus": "unlisted"},
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    logging.info("Uploading video to YouTube...")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    video_id = response.get("id")
    logging.info("YouTube upload complete: %s", video_id)
    return video_id
