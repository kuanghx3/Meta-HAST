import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from tqdm import tqdm
from args import dev
import myfunctions as fn
import pandas as pd

class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout=0, alpha=0.2):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse metrics
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat

class Temporal(nn.Module):
    def __init__(self, args):
        super(Temporal, self).__init__()
        self.nodes = args.nodes
        self.seq_len = args.seq_len
        self.layer = args.layer
        self.LSTMshort1 = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=1, batch_first=True)
        self.LSTMshort2 = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=1, batch_first=True)
        self.LSTMshort3 = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=1, batch_first=True)
        self.LSTMlong = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=2, batch_first=True)
        self.Q_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.K_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.V_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1, _ = self.LSTMshort1(x1)
        x2, _ = self.LSTMshort1(x2)
        x3, _ = self.LSTMshort1(x3)
        y_short = x3[:,-1,0].reshape(-1, self.nodes)
        x_long = torch.cat((x1, x2, x3), dim=1)
        x_long, _ = self.LSTMlong(x_long)
        y_long = x_long[:,-1,0].reshape(-1, self.nodes)
        x_long = x_long.permute(0, 2, 1)
        # --------ATTEN-----------
        Q = self.Q_linear(x_long)
        K = self.K_linear(x_long).permute(0, 2, 1)
        V = self.V_linear(x_long)
        alpha = torch.matmul(Q, K) / K.shape[2]
        alpha = F.softmax(alpha, dim=2)
        ATTEN_out = torch.matmul(alpha, V)
        return ATTEN_out, y_short, y_long

class proposed_Model(nn.Module):
    def __init__(self, args, a_sparse):
        super(proposed_Model, self).__init__()
        self.hidden_size = args.nodes
        self.input_size = args.nodes
        self.output_size = args.nodes
        self.seq_len = args.seq_len
        self.output_size = args.nodes
        self.alpha = args.alpha
        self.layer = args.layer
        self.MLP1 = torch.nn.Sequential(
            torch.nn.Linear(self.seq_len, int(args.MLP_hidden)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden), int(args.MLP_hidden / 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden / 2), self.seq_len))
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq_len, self.seq_len, head_n=4)
        self.gcn = nn.Linear(in_features=self.seq_len, out_features=self.seq_len)
        self.Temporal = Temporal(args)
        self.temp_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.muti_fc = nn.Linear(in_features=2, out_features=1)
        self.MLP_decoder1 = torch.nn.Sequential(
            torch.nn.Linear(self.seq_len, int(args.MLP_hidden)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden), int(args.MLP_hidden / 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden / 2), 1))
        self.MLP_decoder2 = torch.nn.Sequential(
            torch.nn.Linear(self.layer, 8*self.layer),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8*self.layer, 4*self.layer),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4*self.layer, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, occ_atten, temp):
        b, n, s = occ.shape
        occ_atten = self.MLP1(occ_atten)
        occ1 = occ_atten * (1-self.alpha) + occ * self.alpha
        atts_mat = self.gat_lyr(occ)  # dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, occ)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))
        temp_graph = temp.unsqueeze(2).repeat(1,1,self.input_size)
        #temp_graph, _ = self.temp_lstm(temp_graph)
        temp_graph = temp_graph.permute(0, 2, 1)  # (b, n, s)
        occ2 = torch.stack((occ, occ1, occ_conv1, temp_graph), dim=3)
        occ2 = occ2.reshape(b*n, s, self.layer)
        lstm_out, y_short, y_long = self.Temporal(occ2)  # output: (b*n, layer, s)
        y = self.MLP_decoder1(lstm_out)
        y = y.reshape(b, n, self.layer)
        y = self.MLP_decoder2(y)
        y = y.squeeze(2)
        y = (y+y_short+y_long)/3
        return y

def MAML_training(model, nr_meta_loader, lr_meta_loader, hr_meta_loader, args):
    nr_atten = pd.read_csv(args.Atten_path+"no_rain_atten.csv", header=0, index_col=0)
    lr_atten = pd.read_csv(args.Atten_path + "light_rain_atten.csv", header=0, index_col=0)
    hr_atten = pd.read_csv(args.Atten_path + "heavy_rain_atten.csv", header=0, index_col=0)
    nr_atten = torch.Tensor(np.array(nr_atten, dtype=float)).to(dev)
    lr_atten = torch.Tensor(np.array(lr_atten, dtype=float)).to(dev)
    hr_atten = torch.Tensor(np.array(hr_atten, dtype=float)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

    loss_function = torch.nn.MSELoss()
    for _ in tqdm(range(args.meta_epochs)):
        new_model1 = copy.deepcopy(model)
        inner_optim1 = torch.optim.Adam(new_model1.parameters(), lr=args.inner_lr)
        new_model2 = copy.deepcopy(model)
        inner_optim2 = torch.optim.Adam(new_model2.parameters(), lr=args.inner_lr)
        new_model3 = copy.deepcopy(model)
        inner_optim3 = torch.optim.Adam(new_model3.parameters(), lr=args.inner_lr)
        for t, data in enumerate(nr_meta_loader):
            occ, label, temp = data
            occ_s, label_s, temp_s = occ[:int(len(occ)*0.7),:,:], label[:int(len(occ)*0.7),:], temp[:int(len(occ)*0.7),:]
            occ_q, label_q, temp_q = occ[:int(len(occ) * 0.7), :, :], label[:int(len(occ) * 0.7), :], temp[:int(len(occ) * 0.7),:]
            occ_s = occ_s.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_s = torch.matmul(nr_atten, occ_s)  # (batch, node, seq)
            occ_q = occ_q.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_q = torch.matmul(nr_atten, occ_q)  # (batch, node, seq)

            inner_optim1.zero_grad()
            predict = new_model1(occ_s, occ_atten_s, temp_s)
            loss = loss_function(predict, label_s)
            loss.backward()
            inner_optim1.step()
            inner_optim1.zero_grad()

            new_model1_1 = copy.deepcopy(new_model1)
            inner_optim1_1 = torch.optim.Adam(new_model1_1.parameters(), lr=args.inner_lr)
            predict = new_model1_1(occ_q, occ_atten_q, temp_q)
            loss = loss_function(predict, label_q)
            loss.backward()
            inner_optim1_1.step()

            name_to_param = dict(model.named_parameters())
            name_to_param_new1 = dict(new_model1.named_parameters())
            for name, param in new_model1_1.named_parameters():
                cur_grad = (name_to_param_new1[name] - param.data) / args.k * 0.001
                if name_to_param[name].grad is None:
                    name_to_param[name].grad = Variable(torch.zeros(cur_grad.size())).to(dev)
                name_to_param[name].grad.data.add_(cur_grad)
            optimizer.step()
            optimizer.zero_grad()
            inner_optim1_1.zero_grad()

        for t, data in enumerate(lr_meta_loader):
            occ, label, temp = data
            occ_s, label_s, temp_s = occ[:int(len(occ)*0.7),:,:], label[:int(len(occ)*0.7),:], temp[:int(len(occ)*0.7),:]
            occ_q, label_q, temp_q = occ[:int(len(occ) * 0.7), :, :], label[:int(len(occ) * 0.7), :], temp[:int(len(occ) * 0.7),:]
            occ_s = occ_s.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_s = torch.matmul(lr_atten, occ_s)  # (batch, node, seq)
            occ_q = occ_q.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_q = torch.matmul(lr_atten, occ_q)  # (batch, node, seq)

            inner_optim2.zero_grad()
            predict = new_model2(occ_s, occ_atten_s,temp_s)
            loss = loss_function(predict, label_s)
            loss.backward()
            inner_optim2.step()
            inner_optim2.zero_grad()

            new_model2_1 = copy.deepcopy(new_model2)
            inner_optim2_1 = torch.optim.Adam(new_model2_1.parameters(), lr=args.inner_lr)
            predict = new_model2_1(occ_q, occ_atten_q, temp_q)
            loss = loss_function(predict, label_q)
            loss.backward()
            inner_optim2_1.step()

            name_to_param = dict(model.named_parameters())
            name_to_param_new2 = dict(new_model2.named_parameters())
            for name, param in new_model2_1.named_parameters():
                cur_grad = (name_to_param_new2[name] - param.data) / args.k * 0.001
                if name_to_param[name].grad is None:
                    name_to_param[name].grad = Variable(torch.zeros(cur_grad.size())).to(dev)
                name_to_param[name].grad.data.add_(cur_grad)
            optimizer.step()
            optimizer.zero_grad()
            inner_optim2_1.zero_grad()

        for t, data in enumerate(hr_meta_loader):
            occ, label, temp = data
            occ_s, label_s, temp_s = occ[:int(len(occ) * 0.7), :, :], label[:int(len(occ) * 0.7), :], temp[:int(
                len(occ) * 0.7), :]
            occ_q, label_q, temp_q = occ[:int(len(occ) * 0.7), :, :], label[:int(len(occ) * 0.7), :], temp[:int(
                len(occ) * 0.7), :]
            occ_s = occ_s.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_s = torch.matmul(hr_atten, occ_s)  # (batch, node, seq)
            occ_q = occ_q.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten_q = torch.matmul(hr_atten, occ_q)  # (batch, node, seq)

            inner_optim3.zero_grad()
            predict = new_model3(occ_s, occ_atten_s, temp_s)
            loss = loss_function(predict, label_s)
            loss.backward()
            inner_optim3.step()
            inner_optim3.zero_grad()

            new_model3_1 = copy.deepcopy(new_model3)
            inner_optim3_1 = torch.optim.Adam(new_model3_1.parameters(), lr=args.inner_lr)
            predict = new_model3_1(occ_q, occ_atten_q, temp_q)
            loss = loss_function(predict, label_q)
            loss.backward()
            inner_optim3_1.step()

            name_to_param = dict(model.named_parameters())
            name_to_param_new3 = dict(new_model3.named_parameters())
            for name, param in new_model3_1.named_parameters():
                cur_grad = (name_to_param_new3[name] - param.data) / args.k * 0.001
                if name_to_param[name].grad is None:
                    name_to_param[name].grad = Variable(torch.zeros(cur_grad.size())).to(dev)
                name_to_param[name].grad.data.add_(cur_grad)
            optimizer.step()
            optimizer.zero_grad()
            inner_optim3_1.zero_grad()


def reptile_training(model, nr_meta_loader, lr_meta_loader, hr_meta_loader, args):
    nr_atten = pd.read_csv(args.Atten_path+"no_rain_atten.csv", header=0, index_col=0)
    lr_atten = pd.read_csv(args.Atten_path + "light_rain_atten.csv", header=0, index_col=0)
    hr_atten = pd.read_csv(args.Atten_path + "heavy_rain_atten.csv", header=0, index_col=0)
    nr_atten = torch.Tensor(np.array(nr_atten, dtype=float)).to(dev)
    lr_atten = torch.Tensor(np.array(lr_atten, dtype=float)).to(dev)
    hr_atten = torch.Tensor(np.array(hr_atten, dtype=float)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    name_to_param = dict(model.named_parameters())
    loss_function = torch.nn.MSELoss()
    for _ in tqdm(range(args.meta_epochs)):
        new_model = copy.deepcopy(model)
        inner_optim = torch.optim.Adam(new_model.parameters(), lr=args.inner_lr)
        for t, data in enumerate(nr_meta_loader):
            occ, label, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(nr_atten, occ)  # (batch, node, seq)
            inner_optim.zero_grad()
            predict = new_model(occ, occ_atten, temp)
            loss = loss_function(predict, label)
            loss.backward()
            inner_optim.step()
        for t, data in enumerate(lr_meta_loader):
            occ, label, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(lr_atten, occ)  # (batch, node, seq)
            inner_optim.zero_grad()
            predict = new_model(occ, occ_atten, temp)
            loss = loss_function(predict, label)
            loss.backward()
            inner_optim.step()
        for t, data in enumerate(hr_meta_loader):
            occ, label, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(hr_atten, occ)  # (batch, node, seq)
            inner_optim.zero_grad()
            predict = new_model(occ, occ_atten, temp)
            loss = loss_function(predict, label)
            loss.backward()
            inner_optim.step()

        for name, param in new_model.named_parameters():
            cur_grad = ((name_to_param[name].data - param.data) / args.k) * 0.001
            if name_to_param[name].grad is None:
                name_to_param[name].grad = Variable(torch.zeros(cur_grad.size())).to(dev)
            name_to_param[name].grad.data.add_(cur_grad)
        optimizer.step()
        optimizer.zero_grad()

def fine_tuning(model, nr_finetuning_loader, lr_finetuning_loader, hr_finetuning_loader, args, approach):
    nr_atten = pd.read_csv(args.Atten_path + "no_rain_atten.csv", header=0, index_col=0)
    lr_atten = pd.read_csv(args.Atten_path + "light_rain_atten.csv", header=0, index_col=0)
    hr_atten = pd.read_csv(args.Atten_path + "heavy_rain_atten.csv", header=0, index_col=0)
    nr_atten = torch.Tensor(np.array(nr_atten, dtype=float)).to(dev)
    lr_atten = torch.Tensor(np.array(lr_atten, dtype=float)).to(dev)
    hr_atten = torch.Tensor(np.array(hr_atten, dtype=float)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    loss_function = torch.nn.MSELoss()
    losses = []
    for _ in tqdm(range(args.finetuning_epochs)):
        predict = []
        label = []
        for t, data in enumerate(nr_finetuning_loader):
            occ, label_nr, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(nr_atten, occ)  # (batch, node, seq)
            optimizer.zero_grad()
            predict_nr = model(occ, occ_atten, temp)
            predict.append(predict_nr)
            label.append(label_nr)
            loss = loss_function(predict_nr, label_nr)
            loss.backward()
            optimizer.step()
        for t, data in enumerate(lr_finetuning_loader):
            occ, label_lr, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(lr_atten, occ)  # (batch, node, seq)
            optimizer.zero_grad()
            predict_lr = model(occ, occ_atten, temp)
            predict.append(predict_lr)
            label.append(label_lr)
            loss = loss_function(predict_lr, label_lr)
            loss.backward()
            optimizer.step()
        for t, data in enumerate(hr_finetuning_loader):
            occ, label_hr, temp = data
            occ = occ.permute(0, 2, 1)  # (batch, node, seq)
            occ_atten = torch.matmul(hr_atten, occ)  # (batch, node, seq)
            optimizer.zero_grad()
            predict_hr = model(occ, occ_atten, temp)
            predict.append(predict_hr)
            label.append(label_hr)
            loss = loss_function(predict_hr, label_hr)
            loss.backward()
            optimizer.step()

        predict = torch.cat(predict, dim=0)
        label = torch.cat(label, dim=0)
        losses.append(loss_function(predict, label).item())
    # out_loss_df = pd.DataFrame(columns=['train_loss'], data=losses)
    # out_loss_df.to_csv('./result_' + str(args.predict_time) + '/' +approach+"_loss.csv", encoding='gbk')


def test(model, nr_test_loader, lr_test_loader, hr_test_loader, args, approach):
    result = []
    nr_atten = pd.read_csv(args.Atten_path + "no_rain_atten.csv", header=0, index_col=0)
    lr_atten = pd.read_csv(args.Atten_path + "light_rain_atten.csv", header=0, index_col=0)
    hr_atten = pd.read_csv(args.Atten_path + "heavy_rain_atten.csv", header=0, index_col=0)
    nr_atten = torch.Tensor(np.array(nr_atten, dtype=float)).to(dev)
    lr_atten = torch.Tensor(np.array(lr_atten, dtype=float)).to(dev)
    hr_atten = torch.Tensor(np.array(hr_atten, dtype=float)).to(dev)
    model.eval()
    predict = []
    labels = []
    nr_metrics = []
    lr_metrics = []
    hr_metrics = []
    for t, data in enumerate(nr_test_loader):
        occ, label, temp = data
        occ = occ.permute(0, 2, 1)  # (batch, node, seq)
        occ_atten = torch.matmul(nr_atten, occ)  # (batch, node, seq)
        with torch.no_grad():
            predict_nr = model(occ, occ_atten, temp)
        predict.append(predict_nr.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        print("no rain:")
        nr_metrics = fn.get_metrics(predict_nr.cpu().detach().numpy(), label.cpu().detach().numpy())
        result.append(nr_metrics)
    for t, data in enumerate(lr_test_loader):
        occ, label, temp = data
        occ = occ.permute(0, 2, 1)  # (batch, node, seq)
        occ_atten = torch.matmul(lr_atten, occ)  # (batch, node, seq)
        with torch.no_grad():
            predict_lr = model(occ, occ_atten, temp)
        predict.append(predict_lr.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        print("light rain:")
        lr_metrics = fn.get_metrics(predict_lr.cpu().detach().numpy(), label.cpu().detach().numpy())
        result.append(lr_metrics)
    for t, data in enumerate(hr_test_loader):
        occ, label, temp = data
        occ = occ.permute(0, 2, 1)  # (batch, node, seq)
        occ_atten = torch.matmul(hr_atten, occ)  # (batch, node, seq)
        with torch.no_grad():
            predict_hr = model(occ, occ_atten, temp)
        predict.append(predict_hr.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        print("heavy rain:")
        hr_metrics = fn.get_metrics(predict_hr.cpu().detach().numpy(), label.cpu().detach().numpy())
        result.append(hr_metrics)
    predict = np.concatenate(predict, axis=0)  # (2538, 58)
    labels = np.concatenate(labels, axis=0)
    np.save("./draw_plot/reptile_pre_"+str(args.predict_time)+".npy",predict)
    np.save("./draw_plot/label_"+str(args.predict_time)+".npy", labels)
    print("Commercial:")
    C_metrics = fn.get_metrics(predict[:,:29], labels[:,:29])
    result.append(C_metrics)
    print("Office:")
    O_metrics = fn.get_metrics(predict[:, 29:50], labels[:, 29:50])
    result.append(O_metrics)
    print("Residual:")
    R_metrics = fn.get_metrics(predict[:, 50:], labels[:, 50:])
    result.append(R_metrics)
    print("all:")
    all_metrics = fn.get_metrics(predict, labels)
    result.append(all_metrics)
    # pd.DataFrame(data=result, columns=["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2"]).to_csv("./result_"+str(args.predict_time)+"/"+approach+".csv")
    return nr_metrics, lr_metrics, hr_metrics, all_metrics
