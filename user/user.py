"""
获取，更新用户的信息
"""
from pymongo import MongoClient
import redis
from uuid import uuid1
import time
import json

"""
### redis
{
user_id:"id",
user_background:{}
last_entity:[]
last_conversation_time:int(time):
}

userid_conversation_id:""

### monodb 存储对话记录
{user_id:,conversion_id:,from:user/bot,message:"",create_time,entity:[],attention:[]}
"""

HOST = "localhost"
CNVERSION_EXPERID_TIME = 60 * 10  # 10分钟，连续10分钟没有通信，意味着会话结束


class MessageManager:
    def __init__(self):
        self.client = MongoClient(host=HOST)
        self.m = self.client["toutiao"]["dialogue"]
        self.r = redis.Redis(host=HOST, port=6379, db=10)

    def last_entity(self, user_id):
        """最近一次的entity"""
        return json.loads(self.r.hget(user_id, "entity"))

    def gen_conversation_id(self):
        return uuid1().hex

    def bot_message_pipeline(self, user_id, message):
        """保存机器人的回复记录"""
        conversation_id_key = "{}_conversion_id".format(user_id)
        conversation_id = self.user_exist(conversation_id_key)
        if conversation_id:
            # 更新conversation_id的过期时间
            self.r.expire(conversation_id_key, CNVERSION_EXPERID_TIME)
            data = {"user_id": user_id,
                    "conversation_id": conversation_id,
                    "from": "bot",
                    "message": message,
                    "create_time": int(time.time()),
                    }
            self.m.save(data)

        else:
            raise ValueError("没有会话id，但是机器人尝试回复....")

    def user_message_pipeline(self, user_id, message, create_time, attention, entity=[]):
        # 确定用户相关的信息
        # 1. 用户是否存在
        # 2.1 用户存在，返回用户的最近的entity，存入最近的对话
        # 3.1 判断是否为新的对话，如果是新对话，开启新的回话，update用户的对话信息
        # 3.2 如果不是新的对话，update用户的对话信息
        # 3. 更新用户的基本信息
        # 4  返回用户相关信息
        # 5. 调用预测接口，发来对话的结构

        # 要保存的data数据，缺少conversation_id
        data = {
            "user_id": user_id,
            "from": "user",
            "message": message,
            "create_time": create_time,
            "entity": json.dumps(entity),
            "attention": attention,
        }

        conversation_id_key = "{}_conversion_id".format(user_id)
        conversation_id = self.user_exist(conversation_id_key)
        print("conversation_id", conversation_id)
        if conversation_id:
            if entity:
                # 更新当前用户的 last_entity
                self.r.hset(user_id, "last_entity", json.dumps(entity))
            # 更新最后的对话时间
            self.r.hset(user_id, "last_conversion_time", create_time)
            # 设置conversation id的过期时间
            self.r.expire(conversation_id_key, CNVERSION_EXPERID_TIME)

            # 保存聊天记录到mongodb中
            data["conversation_id"] = conversation_id

            self.m.save(data)
            print("mongodb 保存数据成功")

        else:
            # 不存在
            user_basic_info = {
                "user_id": user_id,
                "last_conversion_time": create_time,
                "last_entity": json.dumps(entity)
            }
            self.r.hmset(user_id, user_basic_info)
            print("redis存入 user_basic_info success")
            conversation_id = self.gen_conversation_id()
            print("生成conversation_id", conversation_id)

            # 设置会话的id
            self.r.set(conversation_id_key, conversation_id, ex=CNVERSION_EXPERID_TIME)
            # 保存聊天记录到mongodb中
            data["conversation_id"] = conversation_id
            self.m.save(data)
            print("mongodb 保存数据成功")

    def user_exist(self, conversation_id_key):
        """
        判断用户是否存在
        :param user_id:用户id
        :return:
        """
        conversation_id = self.r.get(conversation_id_key)
        if conversation_id:
            conversation_id = conversation_id.decode()
        print("load conversation_id", conversation_id)
        return conversation_id
