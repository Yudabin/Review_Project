from bs4 import BeautifulSoup
import urllib.request

def image_poster(url):
  global img_url
  image_url = url.split('basic.nhn?c')[0] + 'photoViewPopup.nhn?movieC' + url.split('basic.nhn?c')[1]
  # print(image_url)
  html = urllib.request.urlopen(image_url)
  soup = BeautifulSoup(html, 'html5lib')
  # img_selector = soup.find_all('img')
  imgUrl = soup.find("img")["src"]
  urllib.request.urlretrieve(imgUrl, './static/images/movieposter.jpg')