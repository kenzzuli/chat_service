import torch

"""配置文件"""
from chatbot.sentence2sequence import load_model

by_char = True  # 决定整个模型是按词还是按字
# 如果更改by_char，请删去model/dnn/recall/中的所有search_index，重新生成。

# 原始语料
user_dict_path = "./corpus/user_dict/keywords.txt"  # 用户自定义词典，jieba分词使用
q_a_pair_path = "./corpus/question_answer_pair/QA.xlsx"  # qa问答对
q_path = "./corpus/dnn/recall/Q.txt"  # 召回要使用的q
a_path = "./corpus/dnn/recall/A.txt"  # q
qa_excel_path = "./corpus/dnn/recall/QA.xlsx"  # excel存储的qa
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
chatbot_input_path = chatbot_input_by_char_path if by_char else chatbot_input_by_word_path
chatbot_target_path = chatbot_target_by_char_path if by_char else chatbot_target_by_word_path

# sen2seq
# 路径
# 以词为单位的s2s模型路径
s2s_input_by_word_path = "model/chatbot/s2s_input_by_word.pkl"
s2s_target_by_word_path = "model/chatbot/s2s_target_by_word.pkl"
# 以字为单位的s2s模型路径
s2s_input_by_char_path = "model/chatbot/s2s_input_by_char.pkl"
s2s_target_by_char_path = "model/chatbot/s2s_target_by_char.pkl"

s2s_input_path = s2s_input_by_char_path if by_char else s2s_input_by_word_path
s2s_target_path = s2s_target_by_char_path if by_char else s2s_target_by_word_path
# 模型


s2s_input = load_model(s2s_input_path)
s2s_target = load_model(s2s_target_path)

# chatbot dataloader相关
seq_len_by_char = 40
seq_len_by_word = 30
seq_len = seq_len_by_char if by_char else seq_len_by_word

train_batch_size = 128
test_batch_size = 128
drop_last = True

# chatbot 模型参数
embedding_dim = 256
padding_index = s2s_input.dict[s2s_input.PAD_TAG]
sos_index = s2s_input.dict[s2s_input.SOS_TAG]
eos_index = s2s_input.dict[s2s_input.EOS_TAG]
encoder_hidden_size = 128
encoder_num_layers = 3
encoder_batch_first = False
encoder_drop_out = 0.1
encoder_bidirectional = False
encoder_num_directions = 2 if encoder_bidirectional else 1

decoder_hidden_size = 128
decoder_num_layers = 3
decoder_batch_first = False
decoder_drop_out = 0.1
decoder_bidirectional = False
decoder_num_directions = 2 if decoder_bidirectional else 1

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.3

chatbot_model_by_char_path = "./model/chatbot/chatbot_model_by_char.pkl"
chatbot_model_by_word_path = "./model/chatbot/chatbot_model_by_word.pkl"
chatbot_optimizer_by_char_path = "./model/chatbot/chatbot_optimizer_by_char.pkl"
chatbot_optimizer_by_word_path = "./model/chatbot/chatbot_optimizer_by_word.pkl"
chatbot_model_path = chatbot_model_by_char_path if by_char else chatbot_model_by_word_path
chatbot_optimizer_path = chatbot_optimizer_by_char_path if by_char else chatbot_optimizer_by_word_path

EPOCHS = 1

# 束集搜索宽度
beam_width = 10
# 梯度裁剪
clip = 0.01

# recall相关
# qa dict
qa_path = "./corpus/dnn/recall/qa_dict.json"
search_index_tfidf_path = "./model/dnn/recall/search_index_tfidf.pickle"
search_index_bm25_path = "./model/dnn/recall/search_index_bm25.pickle"
search_index_fasttext_path = "./model/dnn/recall/search_index_fasttext.pickle"
recall_nums = 15
recall_clusters = 10
recall_corpus_by_char = "./corpus/dnn/sort/recall_corpus_by_char.txt"
recall_corpus_by_word = "./corpus/dnn/sort/recall_corpus_by_word.txt"
recall_corpus = recall_corpus_by_char if by_char else recall_corpus_by_word
fasttext_vectorizer_by_char_path = "./model/dnn/recall/fasttext_vectorizer_by_char.model"
fasttext_vectorizer_by_word_path = "./model/dnn/recall/fasttext_vectorizer_by_word.model"
fasttext_vectorizer_path = fasttext_vectorizer_by_char_path if by_char else fasttext_vectorizer_by_word_path

# sort相关
sort_ws_path = "./model/dnn/sort/sort_ws.pickle"  # word2sequence model
sort_q_path = "./corpus/dnn/sort/q_cut_by_char.txt"
sort_similarity_path = "./corpus/dnn/sort/sim_label.txt"  # 衡量两个问题是否相似的标签
sort_similar_q_path = "./corpus/dnn/sort/sim_q_cut_by_char.txt"
sort_ws = load_model(sort_ws_path)  # 加载w2s model
sort_ws_padding_index = sort_ws.dict[sort_ws.PAD_TAG]
sort_batch_size = 128
sort_embedding_dim = 300
sort_hidden_size = 128
sort_num_layers = 3
sort_batch_first = True
sort_dropout = 0.1
sort_bidirectional = False
sort_num_directions = 2 if sort_bidirectional else 1
sort_pooling_stride = 2
sort_pooling_kernel_size = 2
sort_linear_size = 128

dnn_model_path = "./model/dnn/sort/model.pkl"
dnn_optimizer_path = "./model/dnn/sort/optimizer.pkl"
sort_dnn_epoch = 10
sort_threshold = 0.95
