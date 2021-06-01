"""
完整整个模型的逻辑
先分类，根据分类结果选择是qa还是chat
然后调用相应的模型返回答案
"""
from classify.classify import Classify
from chatbot.chatbot import Chatbot
from dnn.sort.sort import DnnSort
from dnn.recall.recall import Recall
from user.user import MessageManager
from chatbot_grpc import chatbot_pb2_grpc
from chatbot_grpc import chatbot_pb2
import time
import grpc
from concurrent import futures


class ChatServicer(chatbot_pb2_grpc.ChatBotServiceServicer):

    def __init__(self):
        # 提前加载各种模型
        self.classify = Classify()
        self.recall = Recall()
        self.sort = DnnSort()
        self.chatbot = Chatbot()
        self.message_manager = MessageManager()

    def Chatbot(self, request, context):
        user_id = request.user_id
        message = request.user_message
        create_time = request.create_time
        attention, prob = self.classify.predict(message)
        if attention == "QA":
            # 实现对对话数据的保存
            self.message_manager.user_message_pipeline(user_id, message, create_time, attention,
                                                       entity=message["entity"])
            recall_list, entity = self.recall.predict(message)
            user_response = self.sort.predict(message, recall_list)

        else:
            # 实现对对话数据的保存
            self.message_manager.user_message_pipeline(user_id, message, create_time, attention,
                                                       entity=message["entity"])
            user_response = self.chatbot.predict(message)

        self.message_manager.bot_message_pipeline(user_id, user_response)

        create_time = int(time.time())
        return chatbot_pb2.ResponsedMessage(user_response=user_response, create_time=create_time)


def server():
    # 多线程服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 注册本地服务
    chatbot_pb2_grpc.add_ChatBotServiceServicer_to_server(ChatServicer(), server)
    # 监听端口
    server.add_insecure_port("[::]:9999")
    # 开始接收请求进行服务
    server.start()
    # 使用 ctrl+c 可以退出服务
    try:
        time.sleep(1000)
    except KeyboardInterrupt:
        server.stop(0)


# todo 这里客户端没成功，以后再看吧
def client():
    with grpc.insecure_channel('localhost:9999') as channel:
        stub = chatbot_pb2_grpc.ChatBotServiceStub(channel)
        response = stub.GetMsg(chatbot_pb2.MsgRequest(name='world'))
    print("Client received: " + response.msg)


if __name__ == '__main__':
    server()
    # client()
