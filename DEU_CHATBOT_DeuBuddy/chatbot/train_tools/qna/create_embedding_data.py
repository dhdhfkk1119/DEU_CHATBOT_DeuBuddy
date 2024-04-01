import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer
'''
train_file = 'train_data.xlsx'
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

df = pd.read_excel(train_file)
df['embedding_vector'] = df['질문(Query)'].progress_map(lambda x : model.encode(x))
df.to_excel("train_data_embedding.xlsx", index=True)

embedding_data = torch.tensor(df['embedding_vector'].tolist())
torch.save(embedding_data, 'embedding_data.pt')
'''

class create_embedding_data:
    def __init__(self, preprocess, df):
        # 텍스트 전처리기
        self.p = preprocess

        # 질문 데이터프레임
        self.df = df

        # pre-trained SBERT
        self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    def create_pt_file(self):
        # 질문 목록 리스트
        target_df = list(self.df['질문(Query)'])

        # 형태소 분석
        for i in range(len(target_df)):
            sentence = target_df[i]
            pos = self.p.pos(sentence)
            keywords = self.p.get_keywords(pos, without_tag=True)
            temp = ""
            for k in keywords:
                temp += str(k)
            target_df[i] = temp

        self.df['질문 전처리'] = target_df
        self.df['embedding_vector'] = self.df['질문 전처리'].progress_map(lambda x : self.model.encode(x))
        self.df.to_excel("/Users/chanmin/DEU_CHATBOT_deudeu/chatbot/train_tools/qna/train_data_embedding.xlsx", index=True) # 절대 경로

        # 아래의 오류를 해결하기 위해서 리스트 대신 numpy 배열로 변환한 후 텐서로 변환
        embedding_data = np.array(self.df['embedding_vector'].tolist())
        embedding_data_tensor = torch.tensor(embedding_data)
        torch.save(embedding_data_tensor, '/Users/chanmin/DEU_CHATBOT_deudeu/chatbot/train_tools/qna/embedding_data.pt') # 절대 경로

        # embedding_data = torch.tensor(self.df['embedding_vector'].tolist())
        # torch.save(embedding_data, '/Users/chanmin/PycharmProjects/chatbot/train_tools/qna/embedding_data.pt')
        # 오류 발생 : UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. ~~
        # 리스트로부터 텐서를 생성하는 과정이 매우 느리다는 것을 알려주고 있음.일반적으로 이 오류는 특별히 큰 크기의 데이터를 처리할 때 발생할 수 있다.

