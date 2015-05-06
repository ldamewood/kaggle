# -*- coding: utf-8 -*-

import requests
import yaml

from lxml import html
from os.path import join, expanduser, basename
import pandas as pd

class Kaggle:
    
    __credentials__ = join(expanduser('~'), '.kaggle.yml')

    @classmethod
    def get_file(cls, remote, local, chunk_size = 512 * 1024):
        """
        Download a file from kaggle.com
        """
        r = requests.get(remote)
        r = requests.post(r.url, data = yaml.load(open(cls.__credentials__)))
        with open(local, 'w') as f:
            for chunk in r.iter_content(chunk_size = chunk_size):
                if chunk:
                    f.write(chunk)
    
    @classmethod
    def submit_file(cls, local, remote):
        """
        Make a submission to kaggle.com
        """
        pass

class KaggleCompetition:
    __full_name__ = None
    __short_name__ = None
    __input__ = []
    __data__ = []
    __data_path__ = 'data'
    
    @classmethod
    def remote_files(cls):
        url = 'https://www.kaggle.com/c/{}/data'.format(cls.__full_name__)
        xml = html.fromstring(requests.get(url).text)
        files = map(lambda x: 'https://www.kaggle.com{}'.format(x),
                    xml.xpath('//table[@id="data-files"]/tbody/tr/td/a/@href'))
        return files
    
    @classmethod
    def get_data(cls):
        for rfile in cls.remote_files():
            Kaggle.get_file(rfile, join(cls.__data_path__, basename(rfile)))

    @classmethod
    def leaderboard(cls):
        url = 'https://www.kaggle.com/c/{}/leaderboard'.format(cls.__full_name__)
        xml = html.fromstring(requests.get(url).text)
        names = map(lambda x: x.strip(), xml.xpath('//td/a[contains(@class, "team-link")]/text()'))
        scores = map(float, xml.xpath('//td/abbr[@class="score"]/text()'))
        lb = pd.DataFrame({'Name': names, 'Score': scores})
        lb.index += 1
        return lb
    
    