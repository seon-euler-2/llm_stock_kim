import streamlit as st
import praw
import ssl
import re
import requests
import pandas as pd
from collections import Counter, defaultdict
import yfinance as yf
import io
import warnings
import contextlib
from urllib3.exceptions import InsecureRequestWarning

from curl_cffi.requests import Session
import yfinance as yf
import pandas as pd
import numpy as np
import ffn
from langchain_core.tools import tool
# from duckduckgo_search import DDGS
from openai import OpenAI
import os
import html
import urllib.parse
import requests
from bs4 import BeautifulSoup
from googlesearch import search  # google 라이브러리 사용
import requests
from bs4 import BeautifulSoup
import os
from pytrends.request import TrendReq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import time
from collections import Counter
import praw
from openai import OpenAI
from dotenv import load_dotenv
import os
# .env 파일에서 환경 변수 로드
load_dotenv()
import reddit_data_collector as rdc
import pandas as pd
import os
import numpy as np
import openai
import json 
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from curl_cffi.requests import Session
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


# SSL 인증서 우회
ssl._create_default_https_context = ssl._create_unverified_context
old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
    requests.Session.merge_environment_settings = merge_environment_settings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings
        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

session = Session(
        impersonate="chrome110",  # optional, mimics Chrome browser
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        verify=False  # optional; disable SSL verify if you're behind a proxy
    )

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
username = os.getenv("REDDIT_USER_AGENT")
password = os.getenv('REDDIT_USER_PASSWORD')
user_agent = os.getenv("REDDIT_USER_AGENT")


print("client_id:", client_id)
print("client_secret:", client_secret)
print("username:", username)
print("password:", password)
print("user_agent:", user_agent)

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
username = os.getenv("REDDIT_USER_AGENT")
password = os.getenv('REDDIT_USER_PASSWORD')
user_agent = os.getenv("REDDIT_USER_AGENT")

# POST 데이터 (password grant_type 사용)
data = {
    'grant_type': 'password',
    'username': username,
    'password': password
}


# POST 요청 헤더
headers = {
    'User-Agent': 'DocumentKey6026'
}

# HTTP Basic 인증 (client_id와 client_secret 사용)
auth = HTTPBasicAuth(client_id, client_secret)

# POST 요청 보내기
response_auth = requests.post(
    'https://www.reddit.com/api/v1/access_token',
    headers=headers,
    data=data,
    auth=auth
)

# 응답 출력
if response_auth.status_code == 200:
    print("Access token:", response_auth.json())
else:
    print(f"Error: {response_auth.status_code}")
    print(response_auth.text)

data_collector = rdc.DataCollector(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)

with no_ssl_verification():
    posts, comments = data_collector.get_data(
        subreddits=["wallstreetbets"],
        post_filter="hot",
        post_limit=100,
        comment_data=True,
        replies_data=True,
        replace_more_limit=0,
        dataframe=True
    )

posts.to_csv("./examples/example_posts.csv", index=False)
comments.to_csv("./examples/example_comments.csv", index=False)