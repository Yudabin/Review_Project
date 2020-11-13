import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
font_location = './wordcloud_file/malgun.ttf'  # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()

plt.rc('font', family=font_name)
def percent_graph2(movie_review):
    b = movie_review
    labelss = ['긍정', '부정']  ## 라벨설정함. 한글이 적용이 안됨!!!
    c = b['score'].value_counts()  ## 빈도

    fig = plt.figure(figsize=(8,8))  ## 캔버스 생성
    fig.set_facecolor('white')  ## 캔버스 배경색을 하얀색으로 설정
    ax = fig.add_subplot()  ## 프레임 생성

    pie = ax.pie(c,  ## 파이차트 출력
                 startangle=90,  ## 시작점을 90도(degree)로 지정
                 counterclock=False,  ## 시계 방향으로 그린다.
                 # autopct=lambda p: '{:.2f}%'.format(p),  ## 퍼센티지 출력
                 wedgeprops=dict(width=0.5),
                 textprops={'fontsize': 20},  # text font size
                 colors = ['yellowgreen', 'orange'])

    total = np.sum(c)  ## 빈도수 총합
    sum_pct = 0  ## 백분율 초기값
    for i, l in enumerate(labelss):
        ang1, ang2 = pie[0][i].theta1, pie[0][i].theta2  ## 각1, 각2
        r = pie[0][i].r  ## 원의 반지름

        x = ((r + 0.5) / 2) * np.cos(np.pi / 180 * ((ang1 + ang2) / 2))  ## 정중앙 x좌표
        y = ((r + 0.5) / 2) * np.sin(np.pi / 180 * ((ang1 + ang2) / 2))  ## 정중앙 y좌표

        if i < len(labelss) - 1:
            sum_pct += float(f'{c[i] / total * 100:.2f}')  ## 백분율을 누적한다.
            ax.text(x, y, f'긍정\n {c[i] / total * 100:.2f}%', ha='center', va='center', size=20, color='white',
                    weight='bold')  ## 백분율 텍스트 표시
        else:  ## 총합을 100으로 맞추기위해 마지막 백분율은 100에서 백분율 누적값을 빼준다.
            ax.text(x, y, f'부정\n {100 - sum_pct:.2f}%', ha='center', va='center',size=20,color='white',
                    weight='bold')
            # pie.rc('font', family=font_name)
    plt.savefig('./static/images/pos_neg_ratio.png')  # 경로