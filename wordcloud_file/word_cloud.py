
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
import stylecloud
from urllib.request import urlopen
import pandas as pd
from tqdm import tqdm
import json
import stylecloud

import matplotlib.pyplot as plt

from matplotlib import font_manager

font_fname = './wordcloud_file/NanumBarunGothic.ttf'

font_family = font_manager.FontProperties(fname=font_fname).get_name()

plt.rcParams["font.family"] = font_family

def make_words_cloud(total_review):

    # 리뷰데이터 단어 추출 - LTokenize
    word_extractor = WordExtractor(min_frequency=10, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
    word_extractor.train(total_review['review'].values)
    words = word_extractor.extract()

    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    tokenizer = LTokenizer(scores=cohesion_score)

    total_review['tokenized'] = total_review['review'].apply(lambda x: tokenizer.tokenize(x, remove_r=True))
    total_review

    # socor 기준으로 긍정(1) 부정(0) 데이터 나눔.
    positive = total_review[total_review['score'] =='긍정']
    # positive

    negative = total_review[total_review['score'] == '부정']
    # negative

    df = positive['review']
    df2 = negative['review']

    # 긍정 데이터에서 리뷰데이터 style cloud로변환
    mystr =''

    for i in tqdm(range(len(df))):
        mystr +=  str(df.iloc[i]) + '\n'


    with open('positive_final.txt', 'w', encoding='utf-8') as f:
        f.write(mystr)


    stylecloud.gen_stylecloud(file_path='positive_final.txt',
                          icon_name='fas fa-thumbs-up',
                          # palette='colorbrewer.diverging.Spectral_11',
                          # palette='cartocolors.qualitative.Prism_3',
                          # palette='lightbartlein.sequential.Blues10_5',
                          # palette='lightbartlein.diverging.BlueGreen_6',
                          #     palette='cartocolors.diverging.Tropic_2',
                              palette='cartocolors.qualitative.Prism_5',
                          background_color='white',
                          gradient='horizontal',
                          font_path = font_fname,
                          output_name='./static/images/positive.png')

    # 부정 데이터에서  리뷰데이터 style cloud로변환
    mystr2 =''
    for i in tqdm(range(len(df2))):
        mystr2 +=  str(df2.iloc[i]) + '\n'


    with open('negative_final.txt', 'w', encoding='utf-8') as f:
        f.write(mystr2)


    stylecloud.gen_stylecloud(file_path='negative_final.txt',
                          # icon_name='fas fa-tired',
                              icon_name='fas fa-thumbs-down',
                          # palette='colorbrewer.diverging.Spectral_11',
                          palette='cartocolors.sequential.RedOr_5',
                              # palette='matplotlib.Inferno_4',
                          background_color='white',
                          gradient='horizontal',
                          font_path = font_fname,
                          output_name='./static/images/negative.png')

    # 필요없는 단어 버리기
    # stopwords = {'재미','영화'}
    # for word in stopwords:
    #     positive_words_dict.pop(word)
    #     negative_words_dict.pop(word)