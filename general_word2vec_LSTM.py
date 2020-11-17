import gensim
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
import tqdm

class Mydata(object):
    def __init__(self,train=True):
        if train:
            self.data = train_data
            self.label = train_label
        else:
            self.data = test_data
            self.label = test_label
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.label)

class MYLSTM(nn.Module):
    def __init__(self):
        super(MYLSTM,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=True)
        self.lstm = nn.LSTM(128,64,2,bidirectional=True)
        self.linear = nn.Linear(128,8)
    def forward(self,x):
        batch_size,seq_len = x.shape
        x = self.embedding(x)
        h0 = c0 = torch.zeros(4,batch_size,64).cuda()
        x = x.transpose(1,0).contiguous()
        x,(h,c) = self.lstm(x,(h0,c0))
#         print(h)
        x = self.linear(x[15,:,:])
        return x

def process_data(data):
    data_list = []
    for text in data:
        text = [word2ix[word] for word in text]
        if len(text)<=15:
            text.extend((16-len(text))*[word2ix['pad']])
        else:
            text = text[0:15] + [word2ix['pad']]
        data_list.append(text)
    return data_list

def cal_acc(model,flag=True):
    if flag:
        mydata = Mydata()
    else:
        mydata = Mydata(False)
    dataloader = DataLoader(mydata,batch_size=64,shuffle=False)
    total = 0
    for i,(data,label) in enumerate(dataloader):
        data = data.long().cuda()
        label = label.long().cuda()
        results = model(data)
        _,top_indexs = torch.topk(results,1,1)
        top_indexs = top_indexs.transpose(1,0).contiguous()
        total += (top_indexs==label).sum().item()
    if flag:
        train_acc = float(total)/64000
        print("train_acc:{}".format(train_acc))
        return train_acc
    else:
        test_acc = float(total)/16000
        print("test_acc:{}".format(test_acc))
        return test_acc

def main():
    #从文件train_txt中读取训练数据
    with open("train_txt",encoding='utf8') as f1:
        train = []
        train_ = []
        for line in f1:
            line = line.strip()
            line = line.split(" ")
            train.append(line)
    #从文件test_txt中读取测试数据
    with open("test_txt",encoding='utf8') as f2:
        test = []
        test_ = []
        for line in f2:
            line = line.strip()
            line = line.split(" ")
            test.append(line)

    train1 = []
    train1.extend(train)
    train1.extend(test)
    #建立词典
    dictionary = gensim.corpora.Dictionary(train1)

    #建立词语和id之间的关联，{词，对应的id}
    word2ix = dictionary.token2id
    # ix2word = dictionary.id2token
    ix2word = {value:key for key,value in word2ix.items()}
    
    #设置最后一个词
    word2ix['pad'] = len(word2ix)
    ix2word[len(word2ix)-1] = 'pad'
    #定制训练和测试的数据格式
    train_data = process_data(train)
    test_data = process_data(test)
    train_label = [0]*8000+[1]*8000+[2]*8000+[3]*8000+[4]*8000+[5]*8000+[6]*8000+[7]*8000
    test_label = [0]*2000+[1]*2000+[2]*2000+[3]*2000+[4]*2000+[5]*2000+[6]*2000+[7]*2000
    
    train_data = torch.Tensor(train_data)
    test_data = torch.Tensor(test_data)
    train_label = torch.Tensor(train_label)
    test_label = torch.Tensor(test_label)

    torch.manual_seed(1) #为GPU设置随机种子，提高实验效果
    random.seed(1)

    #加载word2vec词向量并嵌入进embedding_matrix
    word2vec_model = gensim.models.Word2Vec.load('word2vec_model_general') 
    embedding_matrix = np.zeros((69084,128))
    for word,i in word2ix.items():
        if word != 'pad':
            embedding_matrix[i] = word2vec_model.wv[word]
        else:
            embedding_matrix[i] = np.array([random.normalvariate(0,1) for j in range(128)],dtype='float32')
        
    embedding_matrix = torch.Tensor(embedding_matrix)

    #定义LSTM
    lstm1 = MYLSTM()
    lstm1.cuda()
    epochs=15
    batch_size=64
    optimizer = torch.optim.Adam(lstm1.parameters(),lr=0.001,weight_decay=0.0002)
    schduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8,last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    mydata = Mydata()
    dataloader = DataLoader(mydata,batch_size=batch_size,shuffle=True)

    #weight_decay 0.0002
    #训练lstm
    loss_list = []
    acc_train = []
    acc_test = []
    for j in range(epochs):
        for i,(data,label) in enumerate(dataloader):
            data = data.long().cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            output = lstm1(data)
            loss = criterion(output,label.view(-1))
            loss.backward()
            optimizer.step()
    #       schduler.step()
        loss_list.append(loss)
        print("第{}epoch训练的loss：{}".format(j,loss))
        train_acc = cal_acc(lstm1)
        test_acc = cal_acc(lstm1,False)
        acc_train.append(train_acc)
        acc_test.append(test_acc)
        schduler.step()
    
    

