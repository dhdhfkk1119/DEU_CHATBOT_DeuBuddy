#
# 챗봇에서 사용하는 사전 파일 생성
#
from chatbot.utils.Preprocess import Preprocess
from keras import preprocessing
import pickle
import pandas as pd

# 말뭉치 데이터 읽어오기 (1)
emotion_data = pd.read_csv('../../변형데이터/감정분류데이터.csv')
purpose_data = pd.read_csv('../../변형데이터/용도별목적대화데이터.csv')
topic_data = pd.read_csv('../../변형데이터/주제별텍스트일상대화데이터.csv')

# 데이터 내의 결측값 제거 (2)
emotion_data.dropna(inplace=True)
purpose_data.dropna(inplace=True)
topic_data.dropna(inplace=True)

text1 = list(emotion_data['Q']) + list(emotion_data['A'])
text2 = list(purpose_data['text'])
text3 = list(topic_data['text'])

corpus_data = text1 + text2 + text3

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성 (3)
p = Preprocess()
dict = []
for c in corpus_data:
    pos = p.pos(c)
    for k in pos:
        dict.append(k[0])

# 사전에 사용될 word2index 생성 (4)
# 사전의 첫 번째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV', num_words=100000)
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index
# print(len(word_index)) # 81248

# 사전 파일 생성 (5)
f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()


