import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
import flask
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

from crawling_image.get_image import image_poster
from wordcloud_file.word_cloud import make_words_cloud
from pos_neg_graph.graph import percent_graph2

from flask import Flask, render_template
from flask_paginate import Pagination, get_page_args

t = Okt()
okt = Okt()

app = Flask(__name__)
# app.template_folder = ''
# users = list(range(100))
#
#
# def get_users(offset=0, per_page=10):
#     return users[offset: offset + per_page]

# 메인 페이지 라우팅
@app.route("/")

@app.route("/index")
def index():
    return flask.render_template('index.html')

# 캐시 삭제
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])

# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

def make_prediction():
    if request.method == 'POST':
        url = request.form['url']
        # image_url = url.split('basic')[0] + 'photoViewPopup' + url.split('basic')[1]
        image_poster(url)

        wordcloud_text = []


        # url = 'https://movie.naver.com/movie/bi/mi/basic.nhn?code=196839'
        review_list = []
        label_list = []
        good_label_list = []
        bad_label_list = []
        good_review_list = []
        bad_review_list = []
        good_score_list = []
        bad_score_list = []
        score_list = []

        ## == 페이지 크롤링 ==
        # request 보냄.
        naver_movie_page_url = url.split('basic')[0] + 'pointWriteFormList' + url.split('basic')[1] + '&page=1'
        response = requests.get(naver_movie_page_url)

        # HTML 텍스트 추출
        html = response.text.strip()

        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(markup=html, features='html5lib')

        # 찾고자 하는 element의 selector
        page_selector = 'div.score_total em'

        # element 찾기
        search_pages = soup.select(page_selector)

        for link in search_pages:
            if ',' in link.text:
                a = link.text
                a1 = a.split(',')[0]
                a2 = a.split(',')[1]
                total_pages = int(a1 + a2)
            else:
                total_pages = int(link.text)
        pages = int(total_pages / 10)
        if pages < 100:
            final_pages = pages
        else:
            final_pages = 100

        ## == 리뷰 크롤링 ==

        for page in range(1, final_pages + 1):
            # URL
            naver_movie_url = url.split('basic')[0] + 'pointWriteFormList' + url.split('basic')[1] + f'&page={page}'

            # request 보냄.
            response = requests.get(naver_movie_url)
            # print(response)

            # HTML 텍스트 추출
            html = response.text.strip()

            # BeautifulSoup 객체 생성
            soup = BeautifulSoup(markup=html, features='html5lib')

            # 찾고자 하는 element의 selector
            ## 리뷰 찾기
            for num in range(0, 11):
                review_selector = f'div.score_reple p span#_filtered_ment_{num}'
                # _filtered_ment_0

                # element 찾기
                search_reviews = soup.select(review_selector)
                for review in search_reviews:
                    review_list.append(review.text.strip())
                    review_list = list(filter(bool, review_list))

        #     review2 = pd.Series(review_list, name='review')
        # movie_review = review2

        # new_sentence = '엄청 재밌어요'
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도',
                     '를', '으로', '자', '에', '와', '한', '하다']
        max_len = 30

        with open('./ml/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

            # for review in movie_review:
            #     review_list.append(review)
        # for new_sentence, review1 in zip(movie_review, review_list):
        for new_sentence in review_list:
            pos = new_sentence
            new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
            new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거

            wordcloud_text.append(new_sentence)
            encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
            pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
            score = float(loaded_model.predict(pad_new))  # 예측

            if (score > 0.5):
                label = f'{int(score * 100)} %'
                label_list.append(label)

                good_label_list.append(label)
                good_review_list.append(pos)

                n = '긍정'
                score_list.append(n)
                good_score_list.append(n)

            else:
                label = f'{int((1 - score) * 100)} %'
                label_list.append(label)
                bad_label_list.append(label)
                bad_review_list.append(pos)

                n = '부정'
                score_list.append(n)
                bad_score_list.append(n)
        result = zip(review_list, label_list, score_list)

        result_len = len(review_list)

        good_result = zip(good_label_list, good_review_list, good_score_list)
        bad_result = zip(bad_label_list, bad_review_list, bad_score_list)

        review = pd.Series(review_list, name='review')
        label1 = pd.Series(label_list, name='label')
        score1 = pd.Series(score_list, name='score')

        final_result = pd.merge(review, label1, left_index=True, right_index=True)
        final_result = pd.merge(final_result, score1, left_index=True, right_index=True)
        # print(len(review), len(label1), len(final_result))

        make_words_cloud(final_result)
        percent_graph2(final_result)
        # 결과 리턴
        total_list = result
        good_list = good_result
        bad_list = bad_result

        final_result.to_csv('final_result_pos.csv', index=False)
        return render_template('index.html', url = url,  final_result=result,score=score_list,
                               image_file='../static/images/movieposter.jpg',
                           good_result = good_result, bad_result=bad_result)

# sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # loaded_model = load_model('./model/best_model.h5')
    loaded_model = load_model('./model/best_model_max_len_35.h5')
    # loaded_model = load_model('./model/best_model_epoch10_batch_size600.h5')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)