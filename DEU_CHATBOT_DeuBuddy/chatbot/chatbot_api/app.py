from flask import Flask, request, jsonify, abort, render_template
import socket
import json

# 챗봇 엔진 서버 정보
host = "127.0.0.1"  # 챗봇 엔진 서버 IP
port = 5050         # 챗봇 엔진 port

# Flask 애플리케이션
app = Flask(__name__)

# 챗봇 엔진 서버와 통신
def get_answer_from_engine(bottype, query):
    # 챗봇 엔진 서버 연결
    mySocket = socket.socket()
    mySocket.connect((host, port))

    # 챗봇 엔진 질의 요청
    json_data = {
        'Query' : query,
        'BotType' : bottype
    }
    message = json.dumps(json_data)
    mySocket.send(message.encode())

    # 챗봇 엔진 답변 출력
    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)

    # 챗봇 엔진 서버 연결 소켓 닫기
    mySocket.close()

    return ret_data

# 챗봇 엔진 query 전송 API
@app.route('/query/<bot_type>', methods=['POST'])
def query(bot_type):
    body = request.get_json()
    try:
        if bot_type == 'NORMAL':
            # 일반 질의응답 API
            ret = get_answer_from_engine(bottype=bot_type, query=body['query'])
            return jsonify(ret)
        else:
            # 정의되지 않은 bot type인 경우 404 에러
            abort(404)

    except Exception as ex:
        # 오류 발생시 500 에러
        abort(500)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)