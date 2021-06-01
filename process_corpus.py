from prepare_corpus.prepare_classify_corpus.builid_classify_corpus import process

# process(by_char=True)
# process(by_char=False)

from prepare_corpus.prepare_chatbot_corpus.chatbot_corpus import prepare_xiaohuangji, replace_emoji

# prepare_xiaohuangji(by_char=False)
# prepare_xiaohuangji(by_char=True)


# sentence = "人工智能+python最喜欢你啦 mua╭(╯ε╰)╮"
# print(replace_emoji(sentence))

# from prepare_corpus.prepare_dnn_corpus.recall_corpus import prepare_recall_corpus
#
# prepare_recall_corpus()

# from prepare_corpus.prepare_user_dict.test_user_dict import test_user_dict
# from prepare_corpus.prepare_question_answer_pair.prepare_q_a_pair import process_question_answer_pair
# from lib import cut
# from lib import stopwords
# test_user_dict()
# process_question_answer_pair()
# sentence = "python难不难，很难吗？啊 果真"
# print(cut(sentence, by_character=False, with_pos=True, use_stopwords=True))
# print(stopwords)

# from prepare_corpus.prepare_dnn_corpus.sort_corpus import extract_and_cut_question
#
# extract_and_cut_question(by_char=True)
# extract_and_cut_question(by_char=False)
