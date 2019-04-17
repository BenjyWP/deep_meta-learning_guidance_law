import gym
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class InnerloopDyna():

    def __init__(self, render = False):
        self.time_st = [0.0]
        self.step = 0
        self.alpha_st = [0.0]
        self.beta_st = [0.0]
        self.Q_st = [0.0]
        self.R_st = [0.0]
        self.ue_st = [0.0]
        self.ur_st = [0.0]
        self.Q_d_st = [0.0]
        self.R_d_st = [0.0]
        self.plot_every = 60
        self.amy_st = [0.0]
        self.amz_st = [0.0]
        self.curr_amy_st = [0.0]
        self.curr_amz_st = [0.0]
        self.time = 0.0
        self.m = 500.0
        self.Ixx = 1.6151
        self.Iyy = 136.2648
        self.Izz = 136.2648
        self.k_f = 0.1483
        self.c_x_0 = 0.03772
        self.c_y_beta = -28.16
        self.c_z_alpha = 28.57
        self.c_n_beta = -10.271
        self.c_m_0_1 = -2.547
        self.c_y_alpha = -2.172
        self.c_y_r = 3.541
        self.c_z_beta = -3.256
        self.c_z_e = -3.697
        self.c_l_beta = 0.43
        self.c_m_0_2 = -0.186
        self.K = 0.00141
        self.k_m = 0.2953
        self.c_m_e = -29.35
        self.c_n_r = -24.60
        self.c_m_0_0 = 0.107
        self.c_l_a = 10.52
        self.rho = 1.225
        self.Vm = 500
        self.rho_sm = 0.9
        self.h = 0.001
        self.dot_alpha = 0.0
        self.dot_beta = 0.0
        self.deg2rad = 1.0# pi / 180;
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

        self.prev_alpha_d = 0.0
        self.prev_beta_d = 0.0
        self.P = 0.0
        self.Q = 0.0
        self.R = 0.0
        self.dot_P = 0.0
        self.dot_Q = 0.0
        self.dot_R = 0.0
        self.u_a = 0.0
        self.u_e = 0.0
        self.u_r = 0.0

        self.k12 = np.array([[5.0,0.0],[0.0,5.0]])#diag([5, 5])
        self.k11 = np.array([[60.0,0.0],[0.0,60.0]])#diag([20, 20])
        self.k_p_1 = np.array([[2.0,0.0],[0.0,2.0]])#diag([2, 2])
        self.k_p_2 = np.array([[0.1,0.0],[0.0,0.1]])#diag([0.1, 0.1])
        self.rho_p1 = 0.9
        self.rho_p2 = 0.9
        self.k21 = np.array([[30.0,0.0],[0.0,30.0]])#diag([10, 10])
        self.k22 = np.array([[2.5,0.0],[0.0,2.5]])#diag([0.5, 0.5])
        self.rho_sm = 0.9
        self.u_qr = np.array([[0.0], [0.0]])
        self.u_qr_c = self.u_qr
        self.dot_u_qr_c = np.array([[0.0], [0.0]])
        self.miu_p = self.u_qr
        self.tao_p = 0.01

        self.k_F_y = self.k_f * self.rho * self.Vm * self.Vm * self.c_y_beta
        self.k_F_z = self.k_f * self.rho * self.Vm * self.Vm * self.c_z_alpha
        self.curr_amy = 0.0
        self.curr_amz = 0.0
        self.render = render
        if render:
            self.setup_plot()

    def sig(self,y,rho):
        out = np.sign(y)*np.power(np.abs(y),rho)
        return out

    def setup_plot(self):
        plt.ion()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.line, = self.ax.plot(np.array(self.time_st), np.array(self.amy_st), 'r-')
        self.linet, = self.ax.plot(np.array(self.time_st), np.array(self.curr_amy_st), 'b-')
        self.line2, = self.ax2.plot(np.array(self.time_st), np.array(self.amz_st), 'r-')
        self.line2t, = self.ax2.plot(np.array(self.time_st), np.array(self.curr_amz_st), 'b-')

        self.ax.set_xlabel("T")
        self.ax.set_ylabel("ACTY")
        self.ax.set_xlim([0, 9])
        self.ax.set_ylim([-220, 220])

        self.ax2.set_xlabel("T")
        self.ax2.set_ylabel("ACTZ")
        self.ax2.set_xlim([0, 9])
        self.ax2.set_ylim([-220, 220])

    def plotFig(self):
        self.line.set_xdata(np.array(self.time_st))
        self.line.set_ydata(np.array(self.amy_st))
        self.linet.set_xdata(np.array(self.time_st))
        self.linet.set_ydata(np.array(self.curr_amy_st))
        self.line2.set_xdata(np.array(self.time_st))
        self.line2.set_ydata(np.array(self.amz_st))
        self.line2t.set_xdata(np.array(self.time_st))
        self.line2t.set_ydata(np.array(self.curr_amz_st))
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except:
            print("draw except")

    def save(self,file_num = 0):
        dyna_save = np.vstack(
            (np.array(self.time_st), np.array(self.alpha_st), np.array(self.beta_st), np.array(self.amy_st),
             np.array(self.amz_st), np.array(self.curr_amy_st), np.array(self.curr_amz_st), np.array(self.Q_st),
             np.array(self.R_st), np.array(self.ue_st), np.array(self.ur_st)))
        np.save('inner_dyna' + str(file_num) + '.npy',dyna_save)

    def update(self,amy,amz,h):

        self.h = h
        self.time = self.time + self.h
        self.step = self.step + 1

        self.alpha_d = amz * self.m / self.k_F_z
        self.beta_d = amy * self.m / self.k_F_y

        F1_alpha = self.k_f * self.rho * self.Vm * self.c_z_alpha * self.alpha * np.cos(self.alpha) / (self.m * np.cos(self.beta))
        G1_alpha = - self.P * np.cos(self.alpha) * np.tan(self.beta) + self.Q - self.R * np.sin(self.alpha) * np.tan(self.beta)
        dot_alpha = F1_alpha + G1_alpha

        self.alpha = self.alpha + dot_alpha * self.h
        F1_beta = self.k_f * self.rho * self.Vm * (
                self.c_y_beta * self.beta * np.cos(self.beta) - self.c_z_alpha * self.alpha * np.sin(self.alpha) * np.sin(
            self.beta)) / self.m
        G1_beta = self.P * np.sin(self.alpha) - self.R * np.cos(self.alpha)
        dot_beta = F1_beta + G1_beta
        self.beta = self.beta + dot_beta * self.h

        G1 = np.array([[1, -np.sin(self.alpha) * np.tan(self.beta)],[0.0, - np.cos(self.alpha)]])
        G1_mat_inv = np.mat(G1).I
        F1_alpha = self.k_f * self.rho * self.Vm * self.c_z_alpha * self.alpha * np.cos(self.alpha) / (self.m * np.cos(self.beta))
        F1_beta = self.k_f * self.rho * self.Vm * (
                    self.c_y_beta * self.beta * np.cos(self.beta) - self.c_z_alpha * self.alpha * np.sin(
                self.alpha) * np.sin(self.beta)) / self.m

        F1 = np.array([[F1_alpha],[F1_beta]])
        z1 = np.array([[self.alpha - self.alpha_d],[self.beta - self.beta_d]])
        dotx1_d =np.array([[0.0],[0.0]])# [alpha_d - prev_alpha_d; beta_d - prev_beta_d] / h;

        u_qr_mat = G1_mat_inv * np.mat(np.mat(-F1) + np.mat(dotx1_d) - np.mat(self.k11) * np.mat(self.sig(z1, self.rho_sm)) - np.mat(self.k12) * np.mat(z1))
        u_qr = np.array(u_qr_mat)

        for i in range(10):
            p = u_qr - self.u_qr_c
            dot_miu_p = np.array(np.mat(self.k_p_1) * np.mat(np.power(np.abs(p), self.rho_p1 + 1)))
            self.miu_p = self.miu_p + dot_miu_p * self.h / 10.0
            diag_sig_p =np.array([[self.sig(p[0], self.rho_p1)[0], 0.0],[0.0, self.sig(p[1], self.rho_p1)[0]]])
            self.dot_u_qr_c = -self.u_qr_c + u_qr - self.tao_p * np.array(np.mat(self.k_p_1) * np.mat(diag_sig_p) * np.mat(self.miu_p)) - self.tao_p * np.array(np.mat(self.k_p_2) * np.mat(self.sig(
                p, self.rho_p2)))
            self.dot_u_qr_c = self.dot_u_qr_c/ self.tao_p
            self.u_qr_c = self.u_qr_c + self.dot_u_qr_c * self.h / 10.0

        self.Q_d = self.u_qr_c[0][0]
        self.R_d = self.u_qr_c[1][0]

        self.P = 0.0
        F2_Q = (-(self.Ixx - self.Izz) * self.P * self.R) / self.Iyy
        F2_R = (-(self.Iyy - self.Ixx) * self.P * self.Q + self.k_m * self.rho * self.Vm * self.Vm * self.c_n_beta * self.beta) / self.Izz
        F2 = np.array([[F2_Q],[F2_R]])
        z2 = np.array([[self.Q - self.Q_d],[self.R - self.R_d]])

        G2 = np.array([[self.k_m * self.rho * self.Vm * self.Vm * self.c_m_e / self.Iyy, 0.0],[0.0, self.k_m * self.rho * self.Vm * self.Vm * self.c_n_r / self.Izz]])
        G2_mat_inv = np.mat(G2).I

        u_delta_mat = G2_mat_inv * np.mat(np.mat(-F2) + np.mat(self.dot_u_qr_c) - np.mat(self.k21) * np.mat(self.sig(z2, self.rho_sm)) - np.mat(self.k22) * np.mat(z2))
        u_delta = np.array(u_delta_mat)
        u_e = u_delta[0][0]
        u_r = u_delta[1][0]
        u_e = max(u_e, -0.697)
        u_e = min(u_e, 0.697)
        u_r = max(u_r, -0.697)
        u_r = min(u_r, 0.697)

        dot_Q = (-(self.Ixx - self.Izz) * self.P * self.R + self.k_m * self.rho * self.Vm * self.Vm * self.c_m_e * u_e) / self.Iyy
        dot_R = (-(
                self.Iyy - self.Ixx) * self.P * self.Q + self.k_m * self.rho * self.Vm * self.Vm * self.c_n_beta * self.beta + self.k_m * self.rho * self.Vm * self.Vm * self.c_n_r * u_r) / self.Izz
        self.Q = self.Q + dot_Q * self.h
        self.R = self.R + dot_R * self.h

        self.curr_amy = self.beta * self.k_F_y / self.m
        self.curr_amz = self.alpha * self.k_F_z / self.m

        if self.step % self.plot_every == 0:
            self.time_st.append(self.time)
            self.alpha_st.append(self.alpha)
            self.beta_st.append(self.beta)
            self.Q_d_st.append(self.Q_d)
            self.R_d_st.append(self.R_d)
            self.ue_st.append(u_e)
            self.ur_st.append(u_r)
            self.Q_st.append(self.Q)
            self.R_st.append(self.R)

            self.amy_st.append(amy)
            self.amz_st.append(amz)
            self.curr_amy_st.append(self.curr_amy)
            self.curr_amz_st.append(self.curr_amz)

        if self.render and self.step % self.plot_every == 0:
            self.plotFig()

        return np.array([self.curr_amy,self.curr_amz])

if __name__ == '__main__':
    Inner = InnerloopDyna(render = True)
    for i in range(500):
        Inner.update(200,200,0.0005)
        print(i)
    Inner.save()
    input()




