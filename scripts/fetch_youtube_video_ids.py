import os
from apiclient.discovery import build
from apiclient.errors import HttpError
import pandas as pd
import tqdm
import baker


DEVELOPER_KEY = os.environ['YOUTUBE_API_KEY']
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def youtube_search(query, pageToken, order):
    youtube = build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY
    )

    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=50,
        type='video',
        videoDuration='short',
        pageToken=pageToken,
        order='relevance',
    ).execute()

    rows = []
    for search_result in search_response.get('items', []):
        rows.append({
            'query' : query,
            'YTID': search_result['id']['videoId'],
            'title': search_result['snippet']['title'],
            'description': search_result['snippet']['description'],
        })

    return search_response.get('nextPageToken', ''), rows

def fetch(query, order, pages=1):
  rows = []
  pageToken = None
  for i in tqdm.tqdm(range(pages)):
      try:
        pageToken, rows_for_page = youtube_search(query, pageToken, order)
        rows.extend(rows_for_page)
      except HttpError as e:
        print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))

  return rows

def download(csv_filename, queries, pages=1):
    order = 'relevance'
    all_rows = []
    for query in queries:
        rows_for_query = fetch(query, order=order, pages=pages)
        all_rows.extend(rows_for_query)

    pd.DataFrame(all_rows).to_csv(csv_filename, index=False)

def main():
    noise = [
        'weird engine sound',
        'engine buzzing/farting sound',
        'car engine noise',
        'Engine knocking',
        'Engine bad sound',
        'Engine Whining Noise',
        'engine blown sounds',
        'engine tick sound',
        'engine Clicking Noise',
        'engine rattle noise',
        'engine Ticking Noise'
    ]
    healthy = [
        'healthy engine sound',
        'Sound of a Healthy Engine'
    ]


    download('noise.csv', queries=noise, pages=1)
    download('healthy.csv', queries=healthy, pages=4)


if __name__ == '__main__':
    main()
