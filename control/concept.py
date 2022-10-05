import torch
from NeuralNetwork import basic_nn
import numpy as np
from control import BASE
from torch import nn
from utils import converter
import copy

# state 81
# encoded state 324
# if skill = 8, encoded skill state = 2592
# action = 2
# skill action = 16
# key, query = 81 -> 324 -> 324
# policy = 324*8 -> 324*8 -> 2
# queue = (324 + 2) -> (324 + 2) -> 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def state_converter(state):
    x = torch.arange(9) * 100 - 400
    y = torch.arange(9) * 100 - 400
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1).to(DEVICE)

    out = torch.exp(-torch.sum(torch.square(xy - state)/10000, -1))
    # out.view(-1).size()
    return out.reshape(-1, 81).squeeze()


def batch_state_converter(state):
    x = torch.arange(9) * 100 - 400
    y = torch.arange(9) * 100 - 400
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1).to(DEVICE)

    out = torch.exp(-torch.sum(torch.square(xy.unsqueeze(0) - state.unsqueeze(1).unsqueeze(1))/10000, -1))
    # out.view(-1).size()
    return out.reshape(-1, 81).squeeze()


class Concept(BASE.BaseControl):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.cont_name = "concept"
        self.key = basic_nn.NValueNN(self.s_l, self.s_l*4, int(self.s_l/3)).to(self.device)
        self.query = basic_nn.NValueNN(self.s_l, self.s_l*4, int(self.s_l/3)).to(self.device)
        self.policy_name = "SAC_conti"

        self.policy_list = []
        self.naf_list = []
        self.upd_queue_list = []
        self.base_queue_list = []

        self.upd_policy = basic_nn.ValueNN(int(self.s_l/3), self.s_l * 4,
                                           self.a_l ** 2 + self.a_l).to(self.device)
        self.upd_queue = basic_nn.ValueNN((int(self.s_l/3) + self.a_l), self.s_l * 4, 1).to(self.device)
        self.base_queue = basic_nn.ValueNN((int(self.s_l/3) + self.a_l), self.s_l * 4, 1).to(self.device)

        network_p = []
        lr_p = []
        weight_decay_p = []

        network_q = []
        lr_q = []
        weight_decay_q = []
        ass_policy = None
        ass_queue = None
        i = 0
        while i < self.sk_n:

            tmp_policy = copy.deepcopy(self.upd_policy)

            assert tmp_policy is not self.upd_policy, "copy error"
            for param in tmp_policy.parameters():
                torch.nn.init.uniform_(param, -0.1, 0.1)
                param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                network_p.append(param)
                lr_p.append(self.l_r*10)
                weight_decay_p.append(0.1)
            self.policy_list.append(tmp_policy)

            tmp_queue = copy.deepcopy(self.upd_queue)
            assert tmp_queue is not self.upd_queue, "copy error"

            for param in tmp_queue.parameters():
                torch.nn.init.uniform_(param, -0.2, 0.2)
                param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                network_q.append(param)
                lr_q.append(self.l_r*10)
                weight_decay_q.append(0.1)
            self.upd_queue_list.append(tmp_queue)

            tmp_naf_policy = converter.NAFPolicy(self.s_l * 4, self.a_l, tmp_policy)
            self.naf_list.append(tmp_naf_policy)

            tmp_base_queue = copy.deepcopy(self.base_queue)
            self.base_queue_list.append(tmp_base_queue)
            i = i + 1
        print("assertion")
        assert self.naf_list[0].policy is self.policy_list[0], "assertion error"

        self.optimizer_p = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in
                                            zip(network_p, lr_p, weight_decay_p)])

        self.optimizer_q = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in
                                            zip(network_q, lr_q, weight_decay_q)])

        self.key_optimizer = torch.optim.SGD(self.key.parameters(), lr=self.l_r, weight_decay=0.1)
        self.query_optimizer = torch.optim.SGD(self.query.parameters(), lr=self.l_r, weight_decay=0.1)
        self.criterion = nn.MSELoss(reduction='mean')

    def encoder_decoder_training(self, *trajectory):
        n_p_s, n_a, n_s, n_r, n_d, skill_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        t_p_s = batch_state_converter(t_p_s)
        t_s = batch_state_converter(t_s)
        # base_batch_batch_matrix = torch.sum(torch.square(self.key(t_p_s).unsqueeze(0) - self.key(t_s).unsqueeze(1)), -1)
        base_batch_batch_matrix = torch.exp(self.key(t_s) @ torch.transpose(self.key(t_p_s), -1, -2))
        random_array = torch.rand(len(t_p_s), 2)*800 - 400
        random_input = batch_state_converter(random_array.to(DEVICE))
        random_key = self.key(random_input)
        print(random_key.size())
        print("rsrs")
        negative = torch.exp(self.key(t_s) @ torch.transpose(random_key, -1, -2))
        # base_batch_batch_matrix = torch.exp(self.key(t_p_s) @ torch.transpose(self.query(t_s), -1, -2))
        diagonal = torch.from_numpy(np.arange(len(n_p_s))).unsqueeze(-1).type(torch.int64).to(self.device)
        output = torch.gather(base_batch_batch_matrix, 1, diagonal).squeeze()
        # bellow = base_batch_batch_matrix.sum(-1)
        neg = negative.sum(-1)
        print("base = ")
        print(output)
        print(neg)
        output = len(t_p_s)*output - neg
        print(output)
        loss = -torch.sum(output)/100

        self.key_optimizer.zero_grad()
        # self.query_optimizer.zero_grad()
        loss.backward()
        for param in self.key.parameters():
            param.grad.data.clamp_(-1, 1)
        """
        for param in self.query.parameters():
            param.grad.data.clamp_(-1, 1)
        """
        self.key_optimizer.step()
        # self.query_optimizer.step()
        return loss

    def reward(self, *trajectory):
        # as far as gain more advantage
        n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
        t_p_s = torch.from_numpy(n_p_s).to(self.device).type(torch.float32)
        t_p_s = batch_state_converter(t_p_s)
        t_s = torch.from_numpy(n_s).to(self.device).type(torch.float32)
        t_s = batch_state_converter(t_s)

        # distance_mat = torch.sum(torch.square(t_p_s.unsqueeze(0) - t_s.unsqueeze(1)), -1)
        with torch.no_grad():
            # distance_mat = torch.sum(torch.square(self.key(t_p_s).unsqueeze(0) - self.query(t_s).unsqueeze(1)), -1)
            distance_mat = torch.sum(torch.square(self.key(t_p_s).unsqueeze(0) - self.key(t_s).unsqueeze(1)), -1)
        i = 0
        traj_l = len(n_p_s)/self.sk_n
        subtract = torch.zeros(len(n_p_s))
        distance = torch.sum(distance_mat, -1)
        while i < len(n_p_s):
            if int((sk_idx[i])*traj_l + n_r[i] + 1) >= int((sk_idx[i]+1)*traj_l):
                subtract[i] = torch.tensor(0)
            else:
                subtract[i] = torch.sum(distance_mat[i][int((sk_idx[i])*traj_l + n_r[i] + 1):int((sk_idx[i]+1)*traj_l)], -1)
            # n_r = time step
            i = i + 1
        constant = 1e+4

        return (distance - subtract.to(self.device))/10

    def get_performance(self):
        return self.buffer.get_performance()

    def simulate(self, index=None, total=None, pretrain=1, traj=None):
        policy = self.naf_list
        self.buffer.simulate(self.policy.action, policy, self.reward, index, tot_idx=total,
                             pretrain=pretrain, traj_l=traj, encoder=self.key)

    def update(self, memory_iter, skill_idx, traj_l):
        i = 0
        loss1 = None
        loss2_ary = None
        self.simulate(index=None, total=skill_idx, pretrain=1, traj=traj_l)
        print("iter start")
        while i < memory_iter:
            i = i + 1
            loss1 = self.encoder_decoder_training(self.buffer.get_dataset())

            loss2_ary = self.policy.update(self.buffer.get_dataset(), policy_list=self.policy_list,
                                           naf_list=self.naf_list,
                                           upd_queue_list=self.upd_queue_list, base_queue_list=self.base_queue_list,
                                           optimizer_p=self.optimizer_p, optimizer_q=self.optimizer_q,
                                           memory_iter=1, encoder=self.key)

        loss_ary = torch.cat((loss2_ary, loss1.unsqueeze(0)), -1)
        return loss_ary, self.naf_list

    def load_model(self, path):

        self.key.load_state_dict(torch.load(path + "/" + self.cont_name + "/" + "key"))
        self.query.load_state_dict(torch.load(path + "/" + self.cont_name + "/" + "query"))
        i = 0
        while i < len(self.policy_list):
            self.policy_list[i].load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "policy" + str(i)))
            self.upd_queue_list[i].load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "queue" + str(i)))
            i = i + 1

    def save_model(self, path):

        torch.save(self.key.state_dict(), path + "/" + self.cont_name + "/" + "key")
        torch.save(self.query.state_dict(), path + "/" + self.cont_name + "/" + "query")
        i = 0
        while i < len(self.policy_list):
            torch.save(self.policy_list[i].state_dict(), path + "/" + self.policy_name + "/" + "policy" + str(i))
            torch.save(self.upd_queue_list[i].state_dict(), path + "/" + self.policy_name + "/" + "queue" + str(i))
            i = i + 1
        return self.key, self.query
