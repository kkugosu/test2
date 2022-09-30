import torch
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Render:
    def __init__(self, policy, index, skill_num, env, key):
        self.policy = policy
        self.index = index
        self.skill_num = skill_num
        self.env = env
        self.key = key

    def rend(self, traj=None):
        n_p_o = self.env.reset()
        t = 0
        local = 0
        total_performance = 0
        fail_time = 0
        cir = 0
        while t < self.skill_num*traj*2:
            # n_a = self.policy.action(n_p_o, self.index, per_one=1)
            print(cir)
            n_a = self.policy.action(n_p_o, index=cir, per_one=1, encoder=self.key)
            if traj is None:
                n_o, n_r, n_d, info = self.env.step(n_a)
            else:
                n_o, n_r, info = self.env.step(n_a)
                n_d = 0
            total_performance = total_performance + n_r
            n_p_o = n_o
            time.sleep(0.05)
            self.env.render()
            t = t + 1

            if traj is None:
                if n_d == 1:
                    print("Episode finished after {} timesteps".format(t))
                    fail_time = fail_time + 1
                    self.env.reset()
            else:
                if t % traj == 0:

                    print("Episode finished after {} timesteps".format(t))
                    fail_time = fail_time + 1
                    cir = cir + 1
                    self.env.reset()
        print("performance = ", total_performance/fail_time)
        self.env.close()
