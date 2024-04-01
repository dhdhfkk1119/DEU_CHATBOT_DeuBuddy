import threading
import json
import pandas as pd
import tensorflow as tf
import torch

from utils.BotServer import BotServer
from utils.Preprocess import Preprocess
from utils.FindAnswer import FindAnswer
from models.intent.IntentModel import IntentModel
from train_tools.qna.create_embedding_data import create_embedding_data


# tensorflow gpu 메모리 할당
# tf는 시작시 메모리를 최대로 할당하기 때문에, 0번 GPU를 2GB 메모리만 사용하도록 설정했음.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# 로그 기능 구현
from logging import handlers
import logging

#log settings
LogFormatter = logging.Formatter('%(asctime)s,%(message)s')

#handler settings
LogHandler = handlers.TimedRotatingFileHandler(filename='./logs/chatbot.log', when='midnight', interval=1, encoding='utf-16')
LogHandler.setFormatter(LogFormatter)
LogHandler.suffix = "%Y%m%d"

#logger set
Logger = logging.getLogger()
Logger.setLevel(logging.ERROR)
Logger.addHandler(LogHandler)

# 전처리 객체 생성
p = Preprocess(word2index_dic='./train_tools/dict/chatbot_dict.bin', userdic='./utils/user_dic.tsv')
print("텍스트 전처리기 로드 완료!")

# 의도 파악 모델
intent = IntentModel(model_name='./models/intent/intent_model.keras',proprocess=p)
print("의도 파악 모델 로드 완료!")

# 엑셀  파일 로드
df = pd.read_excel('./train_tools/qna/train_data.xlsx')
print("엑셀 파일 모델 로드 완료!")

# pt 파일 갱신 및 불러오기
create_embedding_data = create_embedding_data(df=df, preprocess=p)
create_embedding_data.create_pt_file()
embedding_data = torch.load('./train_tools/qna/embedding_data.pt')
print("임베딩 pt 파일 갱신 및 로드 완료")

def to_clinet(conn, addr):
    try:
        # 데이터 수신
        read = conn.recv(2048) # 수신 데이터가 있을 때까지 블로킹
        print("=======================")
        print("Connection from : %s" % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나 오류가 있는 경우
            print("클라이언트 연결 끊어짐")
            exit(0) # 스레드 강제 종료

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]

        # 답변 검색
        f = FindAnswer(df=df, embedding_data=embedding_data, preprocess=p)
        selected_qes, score, answer, imageUrl, query_intent = f.search(query, intent_name)

        if score < 0.5:
            answer = ("죄송해요 질문을 잘 이해하지 못했어요 😢.<br>좀 더 자세한 정보를 알려주시면 더 정확한 도움을 드릴 수 있을 것 같아요."
                      "<br>어떤 내용을 찾으시나요?")
            imageUrl = "없음"
            # 사용자 질문, 예측 의도, 선택된 질문, 선택된 질문 의도, 유사도 점수
            Logger.error(f"{query},{intent_name},{selected_qes},{query_intent},{score}")

        send_json_data_str = {
            "Query" : selected_qes,
            "Answer" : answer,
            "ImageUrl" : imageUrl,
            "Intent" : intent_name
        }
        message = json.dumps(send_json_data_str) # json 객체 문자열로 반환
        conn.send(message.encode()) # 응답 전송

    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    # 봇 서버 동작
    port = 5050
    listen = 1000
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start!")

    while True:
        conn, addr = bot.ready_for_client()
        clinet = threading.Thread(target=to_clinet, args=(
            conn,   # 클라이언트 연결 소켓
            addr,   # 클라이언트 연결 주소 정보
        ))
        clinet.start()

