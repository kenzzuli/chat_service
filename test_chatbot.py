from chatbot.sentence2sequence import run
from chatbot.dataset import ChatbotDataset

# run()

chatbot_dataset = ChatbotDataset()
q, a, q_len, a_len = chatbot_dataset[0]
print(q, a)
q_w = chatbot_dataset.input_s2s.inverse_transform(q)
a_w = chatbot_dataset.target_s2s.inverse_transform(a)
print(q_w, a_w)
print(len(chatbot_dataset))
