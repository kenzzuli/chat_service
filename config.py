"""配置文件"""
from chatbot.sentence2sequence import load_model

# 原始语料
user_dict_path = "./corpus/user_dict/keywords.txt"  # 用户自定义词典，jieba分词使用
q_a_pair_path = "./corpus/question_answer_pair/QA.xlsx"  # qa问答对
stopwords_path = "./corpus/user_dict/stopwords.txt"  # 停用词表
emoji_path = "./corpus/user_dict/emoji.txt"  # 颜表情
xiaohuangji_path = "./corpus/classify/raw_corpus/小黄鸡未分词.conv"
by_hand_path = "./corpus/classify/raw_corpus/手动构造的问题.json"
by_crawl_path = "./corpus/classify/raw_corpus/爬虫抓取的问题.csv"

# classify语料
# 按词切分
classify_corpus_by_word_train_path = "./corpus/classify/processed_corpus/classify_by_word_train.txt"
classify_corpus_by_word_test_path = "./corpus/classify/processed_corpus/classify_by_word_test.txt"
# 按字切分
classify_corpus_by_char_train_path = "./corpus/classify/processed_corpus/classify_by_char_train.txt"
classify_corpus_by_char_test_path = "./corpus/classify/processed_corpus/classify_by_char_test.txt"

# classify模型
# 把词作为特征的模型
classify_model_by_word_path = "model/classify/classify_by_word.model"  # 按词切分语料生成的模型
# 把字作为特征的模型
classify_model_by_char_path = "model/classify/classify_by_char.model"  # 按字切分语料生成的模型

# chatbot语料
# 按词切分
chatbot_input_by_word_path = "./corpus/chatbot/input_by_word.txt"  # 聊天机器人 问 语料 按词切分
chatbot_target_by_word_path = "./corpus/chatbot/target_by_word.txt"  # 聊天机器人 答 语料 按词切分
# 按字切分
chatbot_input_by_char_path = "./corpus/chatbot/input_by_char.txt"  # 聊天机器人 问 语料 按字切分
chatbot_target_by_char_path = "./corpus/chatbot/target_by_char.txt"  # 聊天机器人 答 语料 按字切分

# sen2seq
# 路径
# 以词为单位的s2s模型路径
s2s_input_by_word_path = "model/chatbot/s2s_input_by_word.pkl"
s2s_target_by_word_path = "model/chatbot/s2s_target_by_word.pkl"
# 以字为单位的s2s模型路径
s2s_input_by_char_path = "model/chatbot/s2s_input_by_char.pkl"
s2s_target_by_char_path = "model/chatbot/s2s_target_by_char.pkl"
# 模型
# 以词为单位的s2s模型
s2s_input_by_word = load_model(s2s_input_by_word_path)
s2s_target_by_word = load_model(s2s_target_by_word_path)
# 以字为单位的s2s模型
s2s_input_by_char = load_model(s2s_input_by_char_path)
s2s_target_by_char = load_model(s2s_target_by_char_path)

seq_len = 40
