"""配置文件"""

######语料相关######
user_dict_path = "./corpus/user_dict/keywords.txt"
q_a_pair_path = "./corpus/question_answer_pair/QA.xlsx"
stopwords_path = "./corpus/user_dict/stopwords.txt"
emoji_path = "./corpus/user_dict/emoji.txt"
xiaohuangji_path = "./corpus/classify/raw_corpus/小黄鸡未分词.conv"
by_hand_path = "./corpus/classify/raw_corpus/手动构造的问题.json"
by_crawl_path = "./corpus/classify/raw_corpus/爬虫抓取的问题.csv"

chatbot_input_by_word_path = "./corpus/chatbot/input_by_word.txt"  # 聊天机器人 问 语料 按词切分
chatbot_target_by_word_path = "./corpus/chatbot/target_by_word.txt"  # 聊天机器人 答 语料 按词切分

chatbot_input_by_char_path = "./corpus/chatbot/input_by_char.txt"  # 聊天机器人 问 语料 按字切分
chatbot_target_by_char_path = "./corpus/chatbot/target_by_char.txt"  # 聊天机器人 答 语料 按字切分

classify_corpus_by_word_train_path = "./corpus/classify/processed_corpus/classify_by_word_train.txt"
classify_corpus_by_word_test_path = "./corpus/classify/processed_corpus/classify_by_word_test.txt"

classify_corpus_by_char_train_path = "./corpus/classify/processed_corpus/classify_by_char_train.txt"
classify_corpus_by_char_test_path = "./corpus/classify/processed_corpus/classify_by_char_test.txt"

######分类相关########
# 把词作为特征的模型
classify_model_by_word_path = "model/classify_by_word.model"  # 按词切分语料生成的模型
# 把字作为特征的模型
classify_model_by_char_path = "model/classify_by_char.model"  # 按字切分语料生成的模型
