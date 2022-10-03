from policy import BASE
import torch
import numpy as np
from torch import nn
from NeuralNetwork import basic_nn
from utils import converter
GAMMA = 0.98
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

    return out.reshape(-1, 81).squeeze()


class SACPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def action(self, n_s, policy, index, per_one=1, encoder=None):
        t_s = torch.from_numpy(n_s).type(torch.float32).to(self.device)
        t_s = state_converter(t_s)
        if encoder is None:
            pass
        else:
            t_s = encoder(t_s)
        with torch.no_grad():
            mean, cov, t_a = policy[index].prob(t_s)
        n_a = t_a.cpu().numpy()
        n_a_d = np.sqrt(np.sum(n_a**2))
        n_a = n_a/n_a_d
        return n_a

    def update(self, *trajectory, policy_list, naf_list, upd_queue_list, base_queue_list,
               optimizer_p, optimizer_q, memory_iter=0, encoder=None):
        i = 0
        queue_loss = None
        policy_loss = None
        while i < self.sk_n:
            base_queue_list[i].load_state_dict(upd_queue_list[i].state_dict())
            base_queue_list[i].eval()
            i = i + 1
        i = 0
        if memory_iter != 0:
            self.m_i = memory_iter
        else:
            self.m_i = 1
        while i < self.m_i:

            n_p_s, n_a, n_s, n_r, n_d, sk_idx = np.squeeze(trajectory)
            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_p_s = batch_state_converter(t_p_s)
            if encoder is not None:
                with torch.no_grad():
                    encoded_t_p_s = encoder(t_p_s)
            else:
                encoded_t_p_s = t_p_s
            t_p_s = self.skill_converter(encoded_t_p_s, sk_idx, per_one=0)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_a = self.skill_converter(t_a, sk_idx, per_one=0)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            t_r_u = t_r.unsqueeze(-1)
            t_r = self.skill_converter(t_r_u, sk_idx, per_one=0).squeeze()
            # policy_loss = torch.mean(torch.log(t_p_weight) - t_p_qvalue)
            # we already sampled according to policy

            policy_loss = torch.tensor(0).to(self.device).type(torch.float32)

            skill_id = 3 # seq training
            while skill_id < self.sk_n:
                i = 0
                while i < 10:
                    mean, cov, action = naf_list[skill_id].prob(t_p_s[skill_id])
                    with torch.no_grad():
                        sa_pair = torch.cat((t_p_s[skill_id], action), -1).type(torch.float32)
                        target = base_queue_list[skill_id](sa_pair)
                    diff = (action - mean).unsqueeze(-1)
                    prob = (-1/2)*torch.square(torch.transpose(diff, -1, -2)@torch.linalg.inv(cov)@diff)
                    policy_loss += torch.mean(prob.squeeze() - target.squeeze())
                    i = i + 1
                skill_id = skill_id + 1
            policy_loss = policy_loss/10

            sa_pair = torch.cat((t_p_s, t_a), -1).type(torch.float32)

            skill_id = 3 # seq training
            queue_loss = 0
            while skill_id < self.sk_n:
                t_p_qvalue = upd_queue_list[skill_id](sa_pair[skill_id]).squeeze()
                print("skillid = ", skill_id)
                print(t_p_qvalue)
                t_qvalue = t_r[skill_id]
                print(t_r[skill_id])
                # print(t_qvalue)
                # print(t_p_qvalue)
                queue_loss = queue_loss + self.criterion(t_p_qvalue, t_qvalue)
                skill_id = skill_id + 1

            print("queueloss = ", queue_loss)
            print("policy loss = ", policy_loss)

            # print("queue_loss = ", queue_loss)
            optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)

            i = 3 # seq training
            while i < len(policy_list):
                for param in policy_list[i].parameters():
                    param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                    param.grad.data.clamp_(-1, 1)
                i = i + 1
            optimizer_p.step()

            optimizer_q.zero_grad()
            queue_loss.backward()

            i = 3# seq training

            while i < len(upd_queue_list):
                for param in upd_queue_list[i].parameters():
                    param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
                    param.grad.data.clamp_(-1, 1)
                i = i + 1
            if torch.isnan(queue_loss):
                pass
            else:
                optimizer_q.step()

            i = i + 1
        # print("loss1 = ", policy_loss.squeeze())
        # print("loss2 = ", queue_loss.squeeze())

        return torch.stack((policy_loss.squeeze(), queue_loss.squeeze()))
