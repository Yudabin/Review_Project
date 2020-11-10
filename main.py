import pickle

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
        new_sentence = '엄청 재밌어요'
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도',
                     '를', '으로', '자', '에', '와', '한', '하다']
        max_len = 30

        with open('./ml/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

        # tokenizer = Tokenizer(12725, oov_token = 'OOV') # pickle 사용하기
        # tokenizer.fit_on_texts(new_sentence)

        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(loaded_model.predict(pad_new)) # 예측
        if(score > 0.5):
            label = (score * 100)
            result = '긍정'
        else:
            label = (1 - score) * 100
            result = '부정'

        print(new_sentence)
        word_cloud.make_wordcloud(new_sentence)
        # 결과 리턴
        return render_template('index.html', label=label,result=result)

# sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    loaded_model = load_model('./model/best_model.h5')

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
