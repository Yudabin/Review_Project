import nltk
from konlpy.tag import Okt
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import ImageColorGenerator

def make_wordcloud(new_sentence):
    t = Okt()

    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도',
                 '를','으로','자','에','와','한','하다']

    # new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    # tokens_ko = t.nouns(new_sentence2)
    # new_sentence = [word for word in tokens_ko if not word in stopwords] # 불용어 제거

    ko = nltk.Text(new_sentence)
    data=ko.vocab().most_common(100)
    tmp_data = dict(data)

    wordcloud = WordCloud(font_path='malgun.ttf',
                          relative_scaling=0.2,
                          background_color='white',
                          ).generate_from_frequencies(tmp_data)

    korea_coloring = np.array(Image.open('wordcloud_file/movie.jpg'))
    image_colors = ImageColorGenerator(korea_coloring)

    wordcloud = WordCloud(font_path='malgun.ttf',
                          relative_scaling=0.2,
                          mask=korea_coloring,
                          background_color='white',
                          min_font_size=1,
                          max_font_size=40
                          ).generate_from_frequencies(tmp_data)

    fig = plt.figure(figsize=(10,10))
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
    plt.axis('off')
    # plt.show()
    plt.savefig('./static/images/WCImage.png')

# new_sentence2 = kolaw.open('constitution.txt').read()
# make_wordcloud(new_sentence2)