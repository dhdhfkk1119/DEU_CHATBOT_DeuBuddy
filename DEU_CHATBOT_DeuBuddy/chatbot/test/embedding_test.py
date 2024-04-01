import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from chatbot.models.intent.IntentModel import IntentModel
from chatbot.utils.Preprocess import Preprocess

p = Preprocess(userdic='../utils/user_dic.tsv')
intent = IntentModel(model_name='../models/intent/intent_model.keras', proprocess=p)

def custom_cos_sim(A, B):
    return dot(A, B) / (norm(A)*norm(B))
'''
~~ pt 파일 생성 ~~
from utils.Preprocess import Preprocess
from train_tools.qna.create_embedding_data import create_embedding_data

p = Preprocess(userdic='../utils/user_dic.tsv')

df = pd.read_excel('../train_tools/qna/train_data.xlsx')

create_embedding_data = create_embedding_data(df=df, preprocess=p)
create_embedding_data.create_pt_file()
# embedding_data = torch.load('train_tools/qna/embedding_data.pt')
'''


model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('../train_tools/qna/embedding_data.pt')
df = pd.read_excel('../train_tools/qna/train_data.xlsx')

# 질문 예시 문장
sentence = "산학관 건물 위치가 어디야?"
print("질문 문장 : ",sentence)
sentence = sentence.replace(" ","").upper()
print("공백 제거 & 대문자 변환 문장 : ", sentence)

# 질문 예시 문장 인코딩 후 텐서화
sentence_encode = model.encode(sentence)
sentence_tensor = torch.tensor(sentence_encode)

# 저장한 임베딩 데이터와의 코사인 유사도 측정
cos_sim = util.cos_sim(sentence_tensor, embedding_data)
print(f"가장 높은 코사인 유사도 idx : {int(np.argmax(cos_sim))}" )
predict = intent.predict_class(sentence)
predict_label = intent.labels[predict]

print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)

# 선택된 질문 출력
best_sim_idx = int(np.argmax(cos_sim))
selected_qes = df['질문(Query)'][best_sim_idx]
print(f"선택된 질문 = {selected_qes}")
# 선택된 질문 문장에 대한 인코딩
selected_qes_encode = model.encode(selected_qes)

# 유사도 점수 측정
# score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
score = custom_cos_sim(sentence_tensor, selected_qes_encode)
print(f"선택된 질문과의 유사도 = {score}")

# 답변
answer = df['답변(Answer)'][best_sim_idx]
imageUrl = df['답변 이미지'][best_sim_idx]
print(f"\n답변 : \n{answer}")
print(f"답변 이미지 : {imageUrl}")
