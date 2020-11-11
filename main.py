import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
import flask
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt;

from crawling_image.get_image import image_poster
from wordcloud_file import word_cloud

t = Okt()
okt = Okt()

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])

# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

def make_prediction():
    if request.method == 'POST':
        image_poster('https://movie.naver.com/movie/bi/mi/basic.nhn?code=190010')

        wordcloud_text = []

        url = 'https://movie.naver.com/movie/bi/mi/basic.nhn?code=196839'
        review_list = []
        label_list = []
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

        ## == 리뷰 크롤링 ==

        for page in range(1, pages + 1):
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
            review = pd.Series(review_list, name='review')
        movie_review = review


        # new_sentence = '엄청 재밌어요'
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도',
                     '를', '으로', '자', '에', '와', '한', '하다']
        max_len = 30

        with open('./ml/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

        # for review in movie_review:
        #     review_list.append(review)
        for new_sentence, review in zip(movie_review, review_list):
            new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
            new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거

            wordcloud_text.append(new_sentence)
            encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
            pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
            score = float(loaded_model.predict(pad_new))  # 예측

            if (score > 0.5):
                label = f'{score * 100} 확률로 긍정 리뷰입니다.'
                label_list.append(label)
                # pos_review += review
                # result = '긍정'
            else:
                label = f'{(1 - score) * 100} 확률로 부정 리뷰입니다.'
                label_list.append(label)
                neg_review = review
                # result = '부정'
        result = zip(review_list,label_list)
        review = pd.Series(review_list, name='review')
        label1 = pd.Series(label_list, name='label')

        # final_result = pd.merge(review, label1, left_index=True, right_index=True)
        # print(len(review), len(label1), len(final_result))
        print(len(review_list))
        print(len(label_list))
        word_cloud.make_wordcloud(new_sentence)
        # 결과 리턴
        # html에서 데이터프레임 인식x
        # 데이터프레임을 array(데이터프레임.values)로 변환해서 출력
        print(review_list+label_list)
        return render_template('index.html', label=label, final_result=result,score=label_list)

# sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    loaded_model = load_model('./model/best_model.h5')

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)