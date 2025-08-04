#from curl_cffi.requests import Session
# import yfinance as yf
import pandas as pd
import numpy as np
# import ffn
#from langchain_core.tools import tool
#from duckduckgo_search import DDGS
from openai import OpenAI
import os
import html
import urllib.parse
import requests
from bs4 import BeautifulSoup
#from googlesearch import search  # google 라이브러리 사용
import requests
from bs4 import BeautifulSoup
import os
#from pytrends.request import TrendReq
#from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import contextlib
import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings
@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
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
