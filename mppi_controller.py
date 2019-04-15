import numpy as np
import time

class mpc_controller():
    def __init__(self,
                 env,
                 dyn_model,
                 horizon = 20,
                 cost_fn = None,
                 num_simulated_paths = 1000,):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        #self.num_simulated_paths = 1000
        self.curract = None
        #self.alpha = 0.005 ## step size
        self.sigma = 0.5 ## normal dist variance
        self.coef_lambda = 1
        self.es_gen = 10
        self.cost_std = []


    def init_mppi(self,state):

        #step_ini = 0.1
        theta_l = state[0][3]
        phi_l = state[0][4]

        state = np.repeat(state.reshape([1, -1]), self.num_simulated_paths, axis=0)

        action_iter = np.random.uniform(-1.732, 1.732, (self.num_simulated_paths, 2))

        delta = self.dyn_model.predict(state, action_iter)

        cost = self.cost_fn(state[:, :], action_iter, delta[:, :], theta_l, phi_l, self.horizon)
        act = np.argmin(cost)
        action = action_iter[act] * 0.0

        mean_vector_ini = np.repeat(action,self.horizon)
        self.mean_max = np.repeat(np.array([2, 2]), self.horizon)
        self.mean_min = np.repeat(np.array([-2, -2]), self.horizon)

        self.mean_vec = mean_vector_ini

    def get_ac_mppi(self, state):

        theta_l = state[0][3]
        phi_l = state[0][4]

        Normal_dist = self.sigma * np.random.randn(self.num_simulated_paths, 2 * self.horizon)

        action = self.mean_vec + Normal_dist
        cost = np.zeros([self.num_simulated_paths], dtype=np.float32)
        state = np.repeat(state.reshape([1, -1]), self.num_simulated_paths, axis=0)
        d_theta_l = state[:,8]
        d_phi_l = state[:, 9]

        for i in range(self.horizon):

            action_iter = action[:, i:i + 2]
            # delta is the difference in states per timestamp
            delta = self.dyn_model.predict(state, action_iter)

            d_theta_l = d_theta_l + delta[:,5]
            d_phi_l = d_phi_l + delta[:, 6]
            delta[:, 3] = d_theta_l / 200.0
            delta[:, 4] = d_phi_l / 200.0
            cost = cost + self.cost_fn(state, action_iter, delta, theta_l, phi_l, self.horizon)

            dr = delta[:,0] * 200000

            state = np.hstack((delta[:,0:5] + state[:,0:5], dr.reshape([-1,1]), delta[:,1:5] * 200.0))

        #print(np.std(cost))
        print('traj std:{:.10f}'.format(np.std(cost)))
        cost = (cost - np.min(cost)) / np.std(cost)
        weights = np.exp(- (cost / self.coef_lambda))
        weights = weights/np.sum(weights)
        mean_diff = np.sum(np.multiply(weights,Normal_dist.T), axis = 1)
        self.mean_vec = self.mean_vec + mean_diff
        self.mean_vec = np.minimum(self.mean_vec, self.mean_max)
        self.mean_vec = np.maximum(self.mean_vec, self.mean_min)

        action = self.mean_vec[0:2] * 0.5773

        return action#, min_cost

