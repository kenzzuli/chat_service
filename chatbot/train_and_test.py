import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
from chatbot.dataset import get_dataloader
import config
from chatbot.seq2seq import Seq2Seq
from lib import cut
from config import by_char

# 使用gpu训练
s2s = Seq2Seq().to(config.device)
optimizer = torch.optim.Adam(s2s.parameters())

# 模型加载
if os.path.exists(config.chatbot_model_path) and os.path.exists(config.chatbot_optimizer_path):
    s2s.load_state_dict(torch.load(config.chatbot_model_path, map_location=config.device))
    optimizer.load_state_dict(torch.load(config.chatbot_optimizer_path, map_location=config.device))

train_dataloader, test_dataloader = get_dataloader()


def train():
    for i in range(config.EPOCHS):
        with tqdm(total=len(train_dataloader)) as t:
            for index, (feature, target, feature_length, target_length) in enumerate(train_dataloader):
                t.set_description("Epoch %i/%i" % (i + 1, config.EPOCHS))  # 设置描述
                optimizer.zero_grad()  # 梯度置为0
                feature = feature.to(config.device)
                target = target.to(config.device)
                feature_length = feature_length.to(config.device)
                target_length = target_length.to(config.device)
                y_predict = s2s(feature, feature_length, target)  # [seq_len, batch_size, vocab_size]
                y_predict = y_predict.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
                # 计算损失前，要将 3阶和2阶 变成 2阶和1阶
                y_predict = y_predict.reshape(y_predict.size(0) * y_predict.size(1),
                                              y_predict.size(2))  # [batch_size*seq_len, vocab_size]
                # print(y_predict.size(), y_predict.type())
                target = target.reshape(target.size(0) * target.size(1))  # [batch_size*seq_len]
                # print(target.size(), target.type())
                loss = F.nll_loss(y_predict, target, ignore_index=config.padding_index)
                loss.backward()
                # 增加梯度裁剪
                torch.nn.utils.clip_grad_norm_(s2s.parameters(), config.clip)
                optimizer.step()
                t.set_postfix(loss=loss.item())  # 设置后缀
                t.update(1)  # 手动更新进度条
                if index % 10 == 0:  # 每10个batch保存一次模型
                    torch.save(s2s.state_dict(), config.chatbot_model_path)
                    torch.save(optimizer.state_dict(), config.chatbot_optimizer_path)


def eval():
    s2s.eval()  # 进入评估模式
    loss = 0
    correct = 0
    with tqdm(total=len(test_dataloader)) as t:
        for index, (feature, target, feature_length, target_length) in enumerate(test_dataloader):
            t.set_description("Evaluation")  # 设置描述
            with torch.no_grad():
                feature = feature.to(config.device)
                target = target.to(config.device)
                feature_length = feature_length.to(config.device)
                target_length = target_length.to(config.device)
                y_predict = s2s.evaluate(feature, feature_length)  # [seq_len, batch_size, vocab_size]
                y_predict = y_predict.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
                # 准确率
                pred = y_predict.argmax(dim=-1)  # [batch_size, seq_len]
                # 在decoder中，如果使用while循环，可能得到的结果比target短
                target = target[:, :pred.size(1)]
                # eq会生成[batch_size, seq_len]的布尔矩阵 all是只有一行全为true时，才会返回true，形状为[batch_size]
                correct += pred.eq(target).all(dim=-1).sum()
                # 损失
                # 计算损失前，要将 3阶和2阶 变成 2阶和1阶
                y_predict = y_predict.reshape(y_predict.size(0) * y_predict.size(1),
                                              y_predict.size(2))  # [batch_size*seq_len, vocab_size]
                target = target.reshape(target.size(0) * target.size(1))  # [batch_size*seq_len]
                loss += F.nll_loss(y_predict, target, reduction="sum")
                t.update(1)  # 手动更新进度条
        loss /= len(test_dataloader.dataset)
        acc = 100.0 * correct / len(test_dataloader.dataset)
        print("Avg Loss:{}\tAccuracy:{}%".format(loss, acc))


def predict():
    s2s.eval()
    sentence = input("请输入句子: ")
    # 句子转序列
    sentence = cut(sentence, by_character=by_char)
    s2s_input = config.s2s_input
    s2s_target = config.s2s_target
    seq_len = config.seq_len
    feature = s2s_input.transform(sentence, seq_len)
    # 构造feature和feature——length
    feature = torch.LongTensor(feature).to(config.device).unsqueeze(0)
    feature_length = torch.LongTensor([min(len(sentence), seq_len)]).to(config.device)
    # 预测
    y_predict = s2s.evaluate(feature, feature_length)
    # 转换
    y_predict = y_predict.permute(1, 0, 2)
    # 取最后一个维度的最大值作为预测的结果
    pred = y_predict.argmax(dim=-1)
    # 转成列表
    pred = pred.squeeze().detach().numpy().tolist()
    # 转成句子
    pred = s2s_target.inverse_transform(pred)
    # 拼接
    pred = "".join(pred).split("EOS")[0]
    print("预测结果为:", pred)


def predict_beam_search():
    s2s.eval()
    sentence = input("请输入句子: ")
    # 句子转序列
    sentence = cut(sentence, by_character=by_char)
    s2s_input = config.s2s_input
    s2s_target = config.s2s_target
    seq_len = config.seq_len
    feature = s2s_input.transform(sentence, seq_len)
    # 构造feature和feature——length
    feature = torch.LongTensor(feature).to(config.device).unsqueeze(0)
    feature_length = torch.LongTensor([min(len(sentence), seq_len)]).to(config.device)
    # 预测
    y_predict = s2s.evaluate_beam_search(feature, feature_length)
    # 转成句子
    for i in y_predict:
        print("".join(s2s_target.inverse_transform(i)))
