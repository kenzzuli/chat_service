from prepare_corpus.prepare_classify_corpus.builid_classify_corpus import process

# process(by_char=True)
# process(by_char=False)

from prepare_corpus.prepare_chatbot_corpus.chatbot_corpus import prepare_xiaohuangji, replace_emoji

prepare_xiaohuangji(by_char=False)
prepare_xiaohuangji(by_char=True)


# sentence = "人工智能+python最喜欢你啦 mua╭(╯ε╰)╮"
# print(replace_emoji(sentence))


