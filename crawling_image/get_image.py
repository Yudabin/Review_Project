import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image
import os


def image_poster(title_address):
    url = f'{title_address}'
    req = urllib.request.Request(url)
    res = urllib.request.urlopen(url).read()

    soup = BeautifulSoup(res, 'html.parser')
    soup = soup.find("div", class_="poster")
    # img의 경로를 받아온다
    imgUrl = soup.find("img")["src"]

    # urlretrieve는 다운로드 함수
    # img.alt는 이미지 대체 텍스트
    # urllib.request.urlretrieve(imgUrl, soup.find("img")["alt"] + '.jpg')
    urllib.request.urlretrieve(imgUrl, './static/images/movieposter.jpg')
    plt.show()

