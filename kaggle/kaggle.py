# -*- coding: utf-8 -*-

import requests
import yaml

from os.path import join, dirname, realpath

class Kaggle:
    
    __credentials__ = join(dirname(realpath(__file__)), 'password.yml')

    @classmethod
    def get_file(cls, remote, local, chunk_size = 512 * 1024):
        r = requests.get(remote)
        r = requests.post(r.url, data = yaml.load(open(cls.__credentials__)))
        with open(local, 'w') as f:
            for chunk in r.iter_content(chunk_size = chunk_size):
                if chunk:
                    f.write(chunk)