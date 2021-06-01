"""
训练模型
"""
import config
import torch
from dnn.sort.dataset import dnn_dataloader
from dnn.sort.siamese import Siamese
from torch.optim import Adam
from tqdm import tqdm
import os
import torch.nn.functional as F

model = Siamese().to(config.device)
optimizer = Adam(model.parameters(), lr=0.001)
# 模型加载
if os.path.exists(config.dnn_model_path) and os.path.exists(config.dnn_optimizer_path):
    model.load_state_dict(torch.load(config.dnn_model_path, map_location=config.device))
    optimizer.load_state_dict(torch.load(config.dnn_optimizer_path, map_location=config.device))


def train():
    model.train()
    for i in range(config.sort_dnn_epoch):
        with tqdm(total=len(dnn_dataloader)) as t:
            for index, (q, sim_q, similarity, _, _) in enumerate(dnn_dataloader):
                t.set_description("Epoch %i/%i" % (i + 1, config.sort_dnn_epoch))  # 设置描述
                optimizer.zero_grad()  # 梯度置为0
                q = q.to(config.device)
                sim_q = sim_q.to(config.device)
                similarity = similarity.to(config.device)
                out = model(q, sim_q)  # [ batch_size, 2]
                loss = F.cross_entropy(out, similarity)
                loss.backward()
                # 增加梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()
                t.set_postfix(loss=loss.item())  # 设置后缀
                t.update(1)  # 手动更新进度条
                if index % 10 == 0:  # 每10个batch保存一次模型
                    torch.save(model.state_dict(), config.dnn_model_path)
                    torch.save(optimizer.state_dict(), config.dnn_optimizer_path)


def eval():
    model.eval()  # 进入评估模式
    loss = 0
    correct = 0
    with tqdm(total=len(dnn_dataloader)) as t:
        for index, (q, sim_q, similarity, _, _) in enumerate(dnn_dataloader):
            t.set_description("Evaluation")  # 设置描述
            with torch.no_grad():
                q = q.to(config.device)
                sim_q = sim_q.to(config.device)
                similarity = similarity.to(config.device)
                y_predict = model(q, sim_q)
                # 准确率
                pred = y_predict.argmax(dim=-1)
                correct += pred.eq(similarity).sum()
                loss += F.cross_entropy(y_predict, similarity, reduction="sum")
                t.update(1)  # 手动更新进度条
        loss /= len(dnn_dataloader.dataset)
        acc = 100.0 * correct / len(dnn_dataloader.dataset)
        print("Avg Loss:{:.6f}\tAccuracy:{:.6f}%".format(loss, acc))
