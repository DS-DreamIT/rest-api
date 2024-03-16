from flask import Flask, jsonify, request
from flask_restx import Resource, Api

from konlpy.tag import Okt
from keybert import KeyBERT
import pandas as pd


# 키워드 추출
def keyword_extract(sentences):
    # 전처리(명사화, 불용어 제거)
    okt = Okt()
    stopwords = ['그리고', '것', '그게', '다음', '다행', '최대한', '데리', '오늘', '오후', '일찍', '다시', '무엇', '막상',
                 '다음주', '별일', '일도', '일이', '크게', '방금', '갑자기', '계속', '저번', '심지어', '위해', '꿈속', '끼리',
                 '억지로', '그것', '어제', '일단', '타고', '마침', '사려', '못', '일', '무슨', '찌', '여', '아예', '그냥',
                 '거기', '거의', '바로', '존나', '인지', '역시', '조금', '자꾸', '내야', '하니', '약간', '본격', '달라',
                 '정말', '얼마', '가장', '동안', '보고', '보기', '아무', '가운데', '앞', '뒤', '중간', '밑', '위', '아래',
                 '옆', '대신', '마냥', '다해', '다행', '화가', '강제']

    tokenized_doc = okt.pos(sentences, stem=True)
    array_text = [word[0] for word in tokenized_doc if word[1] == 'Noun' and word[0] not in stopwords]

    bow = []
    kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')

    for j in range(len(array_text)):
        keywords = kw_extractor.extract_keywords(array_text[j])
        bow.append(keywords)

    new_bow = []
    for i in range(0, len(bow)):
        for j in range(len(bow[i])):
            new_bow.append(bow[i][j])

    keyword = pd.DataFrame(new_bow, columns=['keyword', 'weight'])
    result = keyword.groupby('keyword').agg('sum').sort_values('weight', ascending=False).head(20)
    return list(result.index)[:3]  # 키워드 수 조절 가능


# flask
app = Flask(__name__)
api = Api(app)
app.config['DEBUG'] = False             # 배포할 때는 False로!
app.config['JSON_AS_ASCII'] = False     # 한글 깨짐 방지


@api.route('/keyword')
class keywordAPI(Resource):
    def post(self):
        try:
            content = request.json.get('content')
            keywords = keyword_extract(content)
            return jsonify({"keywords": keywords})
        except:
            return jsonify({"keywords": []})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
