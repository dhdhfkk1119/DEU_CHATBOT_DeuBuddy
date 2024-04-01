import tensorflow as tf
from keras.models import Model, load_model
from keras import preprocessing

# 의도 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, proprocess):
        # 의도 클래스별 레이블
        self.labels = {0: "연락처", 1: "장소", 2: "일정", 3: "학식", 4: "기타"}

        # 의도 분류 모델 불러오기
        self.model = load_model(model_name)

        # 챗봇 Preprocess 객체
        self.p = proprocess

    # 의도 클래스 예측
    def predict_class(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장 내 키워드 추출(불용어 제거)
        keyword = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keyword)]

        # 단어 시퀀스 벡터 크기
        # from configure.GlobalParams import MAX_SEQ_LEN

        # 패딩 처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences,maxlen=25, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]