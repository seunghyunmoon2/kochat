import urllib
from abc import ABCMeta, abstractmethod
from urllib.request import urlopen, Request

import bs4

from crawler import config
from crawler.base.base import Crawler


class Searcher(Crawler, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        for key, val in config.SEARCH.items():
            setattr(self, key, val)

    @abstractmethod
    def _make_query(self, *args, **kwargs):
        raise NotImplementedError

    def _naver(self, query, selector):
        url = self.url['naver'] + urllib.parse.quote(query)
        out = bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')
        try:
            return [s.contents for s in out.select(selector)]
        except:
            return None
