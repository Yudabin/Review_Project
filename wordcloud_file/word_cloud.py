import nltk
from konlpy.tag import Okt
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import ImageColorGenerator
from konlpy.tag import Okt
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import ImageColorGenerator
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from collections import Counter
from wordcloud import WordCloud
import random


def make_words_cloud(final_result):
    # 리뷰데이터 단어 추출 - LTokenize
    word_extractor = WordExtractor(min_frequency=10, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
    word_extractor.train(final_result['review'].values)
    words = word_extractor.extract()

    cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
    tokenizer = LTokenizer(scores=cohesion_score)

    final_result['tokenized'] = final_result['review'].apply(lambda x: tokenizer.tokenize(x, remove_r=True))
    final_result

    # socor 기준으로 긍정(1) 부정(0) 데이터 나눔.
    positive = final_result[final_result['score'] =='긍정']
    # positive

    negative = final_result[final_result['score'] == '부정']
    # negative

    # 긍정 데이터에서 토큰화 시킨것 딕트형으로변환
    positive_words = []
    for i in positive['tokenized'].values:
        for k in i:
            positive_words.append(k)
    count = Counter(positive_words)
    positive_words_dict = dict(count)

    # 부정 데이터에서 토큰화 시킨것 딕트형으로변환
    negative_words = []
    for i in negative['tokenized'].values:
        for k in i:
            negative_words.append(k)
    count = Counter(negative_words)
    negative_words_dict = dict(count)

    # 필요없는 단어 버리기
    # stopwords = {'재미', '영화'}
    #
    #
    # for word in stopwords:
    #     positive_words_dict.pop(word)
    #     negative_words_dict.pop(word)

    # 부정 리뷰 워드 클라우드
    def negative(negative_words_dict):
        negative_cloud = WordCloud(
            font_path='malgun.ttf',
            background_color='white', width=500, height=500).generate_from_frequencies(negative_words_dict)

        korea_coloring1 = np.array(
            Image.open('./static/images/bad.png'))
        image_colors = ImageColorGenerator(korea_coloring1)

        def make_color(word, font_size, position, orientation, random_state=None, **kwargs):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            color = 'rgb(%d,%d,%d)' % (r, g, b)
            return color

        wordcloud1 = WordCloud(
            font_path='malgun.ttf',
            relative_scaling=0.2,
            mask=korea_coloring1,
            background_color='white',
            min_font_size=1,
            max_font_size=40
            ).generate_from_frequencies(negative_words_dict)

        wordcloud1.recolor(color_func=make_color)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud1.recolor(color_func=make_color), interpolation='bilinear')
        plt.axis('off')
        # plt.show()
        plt.savefig('./static/images/negative.png')

    # 긍정 리뷰 워드 클라우드
    def positive(positive_words_dict):
        positive_cloud = WordCloud(
            font_path='malgun.ttf',
            background_color='white', width=500, height=500).generate_from_frequencies(positive_words_dict)

        korea_coloring2 = np.array(
            Image.open('./static/images/good.png'))
        image_colors = ImageColorGenerator(korea_coloring2)

        def make_color2(word, font_size, position, orientation, random_state=None, **kwargs):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            color = 'rgb(%d,%d,%d)' % (r, g, b)
            return color

        wordcloud2 = WordCloud(
            font_path='malgun.ttf',
            relative_scaling=0.2,
            mask=korea_coloring2,
            background_color='white',
            min_font_size=1,
            max_font_size=40
            ).generate_from_frequencies(positive_words_dict)

        wordcloud2.recolor(color_func=make_color2)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud2.recolor(color_func=make_color2), interpolation='bilinear')
        plt.axis('off')
        # plt.show()
        plt.savefig('./static/images/positive.png')  # 경로

    positive(positive_words_dict)
    negative(negative_words_dict)
