from chatbot.sentence2sequence import build_all_sen2seq
from chatbot.dataset import ChatbotDataset, get_dataloader


def test_chatbot_dataset():
    """测试dataset"""
    chatbot_dataset = ChatbotDataset(by_char=True)
    q, a, q_len, a_len = chatbot_dataset[1000]
    print(q, a)
    q_w = chatbot_dataset.input_s2s.inverse_transform(q)
    a_w = chatbot_dataset.target_s2s.inverse_transform(a)
    print(q_w, a_w)
    print(len(chatbot_dataset))


def test_chatbot_dataloader():
    """测试dataloader"""
    train_dataloader, test_dataloader = get_dataloader()
    for input, target, input_length, target_length in train_dataloader:
        print(input)
        print(target)
        print(input_length)
        print(target_length)
        break


# 构造所有的sen2seq
# build_all_sen2seq()

test_chatbot_dataset()
test_chatbot_dataloader()
