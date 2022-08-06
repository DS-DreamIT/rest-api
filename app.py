from flask import Flask, jsonify, request
from flask_restx import Resource, Api

import torch
from torch import nn
from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
import numpy as np

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
                 '정말', '얼마']

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
    return list(result.index)[:5]  # 키워드 수 조절 가능


# 감정 분류 모델 로드 준비
device = torch.device("cpu")
bertmodel, vocab = get_pytorch_kobert_model()

# Setting parameters
max_len = 256
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=8,  ##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# 모델 로드
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('./model/8emotions_state_dict_ver3.pt', map_location=device))

# 예측
def predict(model, predict_sentence):
    times = len(predict_sentence) // 500
    sentence = []
    sum_logits = np.zeros(8)

    # 500자씩 문장 나누기
    for i in range(times + 1):
        sentence.append(predict_sentence[500 * i: 500 * (i + 1)])

    for s in sentence:
        data = [s, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=2)

        model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length = valid_length

            out = model(token_ids, valid_length, segment_ids)

            logits = out[0].detach().cpu().numpy()
            # logits 합
            sum_logits += logits

    # 합해진 감정 logits 정렬
    result = {'공포': sum_logits[0], '분노': sum_logits[1], '불안': sum_logits[2], '슬픔': sum_logits[3], '놀람': sum_logits[4],
              '중립': sum_logits[5], '행복': sum_logits[6], '설렘': sum_logits[7]}
    result = sorted(result.items(), key=lambda item: item[1], reverse=True)

    # 가장 지수가 높은 감정 1~3개 추출
    emotion = []
    emotion.append(result[0][0])
    # 그 전 감정과 차이가 2.5 이상 나면 버림
    for i in range(1, 3):
        if result[i][1] > 0 and (result[i - 1][1] - result[i][1] < 2.5):
            emotion.append(result[i][0])
        else:
            break

    return emotion


# flask
app = Flask(__name__)
api = Api(app)
app.config['DEBUG'] = False             # 배포할 때는 False로!
app.config['JSON_AS_ASCII'] = False     # 한글 깨짐 방지


@api.route('/emotion')
class emotionAPI(Resource):
    def post(self):
        try:
            content = request.json.get('content')
            result = predict(model, content)
            return jsonify({"result": result})
        except:
            return jsonify({"result": []})


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
    app.run(debug=False)
