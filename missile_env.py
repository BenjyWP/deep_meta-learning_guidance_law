"""

Chen Liang, Beihang University
Code accompanying the paper
"Learing to guide: Guidance Law Based on Deep Meta-learning and Model Predictive Path Integral Control"


"""

import numpy as np
import matplotlib
import random

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class missile_env():

    def __init__(self, num_index = 1, render = True):

        self.h = 0.0005
        self.time_true = 0.0
        self.time = 0.0 # iterative time stamp
        self.target_store = np.load('npys/Target_pos.npy')
        self.plot_iter = 1
        self.vmini = 800.0
        self.render = render
        self.file_num = num_index # the index of a run
        self.num_subIter = 10# The sample time of underlying integral is 0.5ms, while control cycle is 5ms
        #angles
        self.theta_missile = 0.0
        self.theta_target = 0.0
        self.theta_los = 0.0
        self.phi_los = 0.0
        self.phi_target = 0.0
        self.phi_missile = 0.0

        self.range = 0.0
        #angluar rates(angle derivatives)
        self.d_phi_los = 0.0
        self.d_theta_los = 0.0
        self.d_theta_missile = 0.0
        self.d_phi_missile = 0.0
        self.d_theta_target = 0.0
        self.d_phi_target = 0.0
        #accelerations
        self.aym = 0
        self.azm = 0
        self.prev_acy = 0.0
        self.prev_acz = 0.0
        self.ayt = 0
        self.azt = 0
        self.ac1Now = 0.0
        self.ac2Now = 0.0

        self.vm = self.vmini
        self.vt = 20.0
        self.xt = 9000
        self.yt = 9000
        self.zt = 9000
        self.xm = 400
        self.ym = 300
        self.zm = 400
        self.T_B = 4.0 #burnout time of thruster

        self.range = np.sqrt((self.xt - self.xm) * (self.xt - self.xm) + (self.yt - self.ym) * (self.yt - self.ym))
        self.deri_range = 0.0

        self.time_plot = np.array([0.0])
        self.ac1_plot = np.array([0.0])
        self.ac2_plot = np.array([0.0])

        self.d_theta_los_plot = np.array([0.0])
        self.d_phi_los_plot = np.array([0.0])
        self.d_theta_missile_plot = np.array([0.0])
        self.d_phi_missile_plot = np.array([0.0])
        self.theta_los_plot = np.array([0.0])
        self.phi_los_plot = np.array([0.0])
        self.range_diff_plot = np.array([0.0])
        self.a_target_plot = np.array([0.0])

        self.xm_plot = np.array([self.xm])
        self.ym_plot = np.array([self.ym])
        self.zm_plot = np.array([self.zm])
        self.xt_plot = np.array([self.xt])
        self.yt_plot = np.array([self.yt])
        self.zt_plot = np.array([self.zt])
        self.range_plot = np.array([self.range])
        self.theta_missile_plot = np.array([self.theta_missile])
        self.phi_missile_plot = np.array([self.phi_missile])
        #target accelerations
        self.targetAcPhaseY = 0.0
        self.targetAcPhaseZ = 0.0
        self.targetAcCysleY = 0.0005
        self.targetAcCysleZ = 0.001

        self.dynTrack = []# current
        if self.render:
            self._setup_plot()


    def _setup_plot(self):
        plt.ion()

        self.fig = plt.figure()

        self.ax = []
        self.lines = []

        for i in range(1,16):
            self.ax.append(self.fig.add_subplot(5, 3, i))


        self.line,  = self.ax[0].plot(self.xm_plot, self.ym_plot, 'r-')
        self.linet, = self.ax[0].plot(self.xt_plot, self.yt_plot, 'b-')
        self.line2,  = self.ax[1].plot(self.ym_plot, self.zm_plot, 'r-')
        self.line2t, = self.ax[1].plot(self.yt_plot, self.zt_plot, 'b-')
        self.line3, = self.ax[2].plot(self.time_plot, self.range_plot, 'r-')
        self.line4, = self.ax[3].plot(self.time_plot, self.ac1_plot, 'r-')
        self.line5, = self.ax[4].plot(self.time_plot, self.ac2_plot, 'r-')
        self.line6, = self.ax[5].plot(self.time_plot, self.d_theta_los_plot, 'r-')
        self.line7, = self.ax[6].plot(self.time_plot, self.d_phi_los_plot, 'r-')
        self.line8, = self.ax[7].plot(self.time_plot, self.theta_los_plot, 'r-')
        self.line9, = self.ax[8].plot(self.time_plot, self.phi_los_plot, 'r-')
        self.line10, = self.ax[9].plot(self.time_plot, self.theta_missile_plot, 'r-')
        self.line11, = self.ax[10].plot(self.time_plot, self.d_theta_missile_plot, 'r-')
        self.line12, = self.ax[11].plot(self.time_plot, self.d_phi_missile_plot, 'r-')
        self.line13, = self.ax[12].plot(self.time_plot, self.phi_missile_plot, 'r-')
        self.line14, = self.ax[13].plot(self.time_plot, self.range_diff_plot, 'r-')
        self.line15, = self.ax[14].plot(self.time_plot, self.a_target_plot, 'r-')

        # plot labels
        self.ax[0].set_xlabel("X")
        self.ax[0].set_ylabel("Y")
        self.ax[0].set_xlim([-2000, 12000])
        self.ax[0].set_ylim([-2000, 12000])

        self.ax[1].set_xlabel("Y")
        self.ax[1].set_ylabel("Z")
        self.ax[1].set_xlim([-2000, 12000])
        self.ax[1].set_ylim([-2000, 12000])

        self.ax[2].set_xlabel("T")
        self.ax[2].set_ylabel("R")
        self.ax[2].set_xlim([0, 10])
        self.ax[2].set_ylim([0, 10000])

        self.ax[3].set_xlabel("T")
        self.ax[3].set_ylabel("ACT1")
        self.ax[3].set_xlim([0, 10])
        self.ax[3].set_ylim([-2, 2])

        self.ax[4].set_xlabel("T")
        self.ax[4].set_ylabel("ACT2")
        self.ax[4].set_xlim([0, 10])
        self.ax[4].set_ylim([-2, 2])

        self.ax[5].set_xlabel("T")
        self.ax[5].set_ylabel("D_theta_l")
        self.ax[5].set_xlim([0, 10])
        self.ax[5].set_ylim([-0.1, 0.1])

        self.ax[6].set_xlabel("T")
        self.ax[6].set_ylabel("D_phi_l")
        self.ax[6].set_xlim([0, 10])
        self.ax[6].set_ylim([-0.2, 0.2])

        self.ax[7].set_xlabel("T")
        self.ax[7].set_ylabel("theta_l")
        self.ax[7].set_xlim([0, 10])
        self.ax[7].set_ylim([-np.pi, np.pi])

        self.ax[8].set_xlabel("T")
        self.ax[8].set_ylabel("phi_l")
        self.ax[8].set_xlim([0, 10])
        self.ax[8].set_ylim([-np.pi, np.pi])

        self.ax[9].set_xlabel("T")
        self.ax[9].set_ylabel("Theta")
        self.ax[9].set_xlim([0, 10])
        self.ax[9].set_ylim([-np.pi /2.0, np.pi /2.0])

        self.ax[10].set_xlabel("T")
        self.ax[10].set_ylabel("D_theta_m")
        self.ax[10].set_xlim([0, 10])
        self.ax[10].set_ylim([-0.2, 0.2])

        self.ax[11].set_xlabel("T")
        self.ax[11].set_ylabel("D_phi_m")
        self.ax[11].set_xlim([0, 10])
        self.ax[11].set_ylim([-0.2, 0.2])

        self.ax[12].set_xlabel("T")
        self.ax[12].set_ylabel("phi_m")
        self.ax[12].set_xlim([0, 10])
        self.ax[12].set_ylim([-np.pi /2.0, np.pi / 2.0])

        self.ax[13].set_xlabel("T")
        self.ax[13].set_ylabel("D_R")
        self.ax[13].set_xlim([0, 10])
        self.ax[13].set_ylim([-1000, 500])

        self.ax[14].set_xlabel("T")
        self.ax[14].set_ylabel("ATY")
        self.ax[14].set_xlim([0, 10])
        self.ax[14].set_ylim([-320, 320])


    def reset_recursive(self):

        self.dynTrack = []
        self.h = 0.0005

        self.time_true = 0.0
        self.time = 0

        self.d_phi_los = 0.0
        self.d_theta_los = 0.0
        self.d_phi_target = 0.0
        self.d_theta_target = 0.0
        self.d_phi_missile = 0.0
        self.d_theta_missile = 0.0

        self.phi_los = 0.0
        self.theta_los = 0.0
        self.phi_missile = 0.0
        self.theta_missile = 0.0
        self.phi_target = 0.0
        self.theta_target = 0.0

        self.range = 0.0
        self.deri_range = 0.0

        self.vm = self.vmini

        self.aym = 0
        self.azm = 0
        self.ayt = 0
        self.azt = 0

        self.theta_target = 0.0
        self.phi_target = -0.0

        self.theta_missile = random.uniform(-1,1) * np.pi / 8.0
        self.phi_missile = random.uniform(-1,1) * np.pi / 8.0


        self.vt = 270
        self.at = 3
        self.xt = 6000.0
        self.yt = 6000.0
        self.zt = 6000.0
        xyz_target = self.target_store[0]
        self.xt = xyz_target[0]
        self.yt = xyz_target[1]
        self.zt = xyz_target[2]

        self.theta_los = -0.6 + random.uniform(-1,1) * 0.1
        self.phi_los = 0.8 + random.uniform(-1,1) * 0.1


        self.range = 3800
        self.T_B = 3.5

        self.zm = self.zt - self.range * np.sin(self.theta_los)
        tempXY = abs(self.range * np.cos(self.theta_los))
        if abs(self.phi_los) < np.pi / 2:
            self.xm = self.xt - tempXY * np.cos(self.phi_los)
            self.ym = self.yt - tempXY * np.sin(self.phi_los)
        else:
            if self.phi_los > 0:
                self.xm = self.xt + tempXY * np.cos(np.pi - self.phi_los)
                self.ym = self.yt - tempXY * np.sin(np.pi - self.phi_los)
            else:
                self.xm = self.xt + tempXY * np.cos(np.pi - abs(self.phi_los))
                self.ym = self.yt + tempXY * np.sin(np.pi - abs(self.phi_los))

        valid_range = np.sqrt((self.xt - self.xm) * (self.xt - self.xm) + (self.yt - self.ym) * (self.yt - self.ym) + (
                self.zt - self.zm) * (self.zt - self.zm)
                        )
        if abs(valid_range - self.range) > 4:
            #print("reset failed with incorrect initial R")
            raise Exception("reset failed with incorrect initial R")

        temptheta_l = np.arctan(
            (self.zt - self.zm) / np.sqrt(np.square(self.xt - self.xm) + np.square(self.yt - self.ym)))
        if self.xt > self.xm:
            tempphi_l = np.arctan((self.yt - self.ym) / (self.xt - self.xm))
        else:
            if self.xt == self.xm:
                tempphi_l = np.pi / 2
            else:
                if self.yt > self.ym:
                    tempphi_l = np.pi - np.arctan((self.yt - self.ym) / (self.xm - self.xt))
                else:
                    tempphi_l = -np.pi + np.arctan((self.yt - self.ym) / (self.xt - self.xm))

        if abs(tempphi_l - self.phi_los) > 0.1 or abs(temptheta_l - self.theta_los) > 0.1:
            #print("reset failed with incorrect initial los")
            raise Exception("reset failed with incorrect initial los")

        self.theta_missile = max(self.theta_missile,-np.pi)
        self.theta_missile = min(self.theta_missile, np.pi)

        self.deri_range = self.vt * np.cos(self.theta_target) * np.cos(self.phi_target) - self.vm * np.cos(self.theta_missile) * np.cos(
            self.phi_missile)

        self.xm_plot = np.array([self.xm])
        self.ym_plot = np.array([self.ym])
        self.zm_plot = np.array([self.zm])
        self.xt_plot = np.array([self.xt])
        self.yt_plot = np.array([self.yt])
        self.zt_plot = np.array([self.zt])
        self.ac1_plot = np.array([0.0])
        self.ac2_plot = np.array([0.0])
        self.time_plot = np.array([0.0])
        self.d_theta_los_plot = np.array([0.0])
        self.d_phi_los_plot = np.array([0.0])
        self.d_theta_missile_plot = np.array([0.0])
        self.d_phi_missile_plot = np.array([0.0])
        self.range_diff_plot = np.array([0.0])
        self.a_target_plot = np.array([0.0])
        self.range_plot = np.array([self.range])
        self.theta_los_plot = np.array([self.theta_los])
        self.phi_los_plot = np.array([self.phi_los])
        self.theta_missile_plot = np.array([self.theta_missile])
        self.phi_missile_plot = np.array([self.phi_missile])

        self._forward_dynamics([0.0, 0.0])
        self._forward_dynamics([0.0, 0.0])

        observation = np.copy(self._state)
        return observation

    def _plotFig(self):
        self.xm_plot = np.vstack((self.xm_plot, np.array([self.xm])))
        self.ym_plot = np.vstack((self.ym_plot, np.array([self.ym])))
        self.xt_plot = np.vstack((self.xt_plot, np.array([self.xt])))
        self.yt_plot = np.vstack((self.yt_plot, np.array([self.yt])))
        self.zm_plot = np.vstack((self.zm_plot, np.array([self.zm])))
        self.zt_plot = np.vstack((self.zt_plot, np.array([self.zt])))
        self.time_plot = np.vstack((self.time_plot, np.array([self.time_true])))
        self.ac1_plot = np.vstack((self.ac1_plot, np.array([self.ac1Now])))
        self.ac2_plot = np.vstack((self.ac2_plot, np.array([self.ac2Now])))
        self.d_theta_los_plot = np.vstack((self.d_theta_los_plot, np.array([self.d_theta_los])))
        self.d_phi_los_plot = np.vstack((self.d_phi_los_plot, np.array([self.d_phi_los])))
        self.d_theta_missile_plot = np.vstack((self.d_theta_missile_plot, np.array([self.d_theta_missile])))
        self.d_phi_missile_plot = np.vstack((self.d_phi_missile_plot, np.array([self.d_phi_missile])))
        self.range_diff_plot = np.vstack((self.range_diff_plot, np.array([self.deri_range])))
        self.a_target_plot = np.vstack((self.a_target_plot, np.array([self.ayt])))
        self.range_plot = np.vstack((self.range_plot, np.array([self.range])))
        self.theta_los_plot = np.vstack((self.theta_los_plot, np.array([self.theta_los])))
        self.phi_los_plot = np.vstack((self.phi_los_plot, np.array([self.phi_los])))
        self.theta_missile_plot = np.vstack((self.theta_missile_plot, np.array([self.theta_missile])))
        self.phi_missile_plot = np.vstack((self.phi_missile_plot, np.array([self.phi_missile])))

        self.line.set_xdata(self.xm_plot)
        self.line.set_ydata(self.ym_plot)
        self.linet.set_xdata(self.xt_plot)
        self.linet.set_ydata(self.yt_plot)
        self.line2.set_xdata(self.ym_plot)
        self.line2.set_ydata(self.zm_plot)
        self.line2t.set_xdata(self.yt_plot)
        self.line2t.set_ydata(self.zt_plot)
        self.line3.set_xdata(self.time_plot)
        self.line3.set_ydata(self.range_plot)
        self.line4.set_xdata(self.time_plot)
        self.line4.set_ydata(self.ac1_plot)
        self.line5.set_xdata(self.time_plot)
        self.line5.set_ydata(self.ac2_plot)
        self.line6.set_xdata(self.time_plot)
        self.line6.set_ydata(self.d_theta_los_plot)
        self.line7.set_xdata(self.time_plot)
        self.line7.set_ydata(self.d_phi_los_plot)
        self.line8.set_xdata(self.time_plot)
        self.line8.set_ydata(self.theta_los_plot)
        self.line9.set_xdata(self.time_plot)
        self.line9.set_ydata(self.phi_los_plot)
        self.line10.set_xdata(self.time_plot)
        self.line10.set_ydata(self.theta_missile_plot)
        self.line11.set_xdata(self.time_plot)
        self.line11.set_ydata(self.d_theta_missile_plot)
        self.line12.set_xdata(self.time_plot)
        self.line12.set_ydata(self.d_phi_missile_plot)
        self.line13.set_xdata(self.time_plot)
        self.line13.set_ydata(self.phi_missile_plot)
        self.line14.set_xdata(self.time_plot)
        self.line14.set_ydata(self.range_diff_plot)
        self.line15.set_xdata(self.time_plot)
        self.line15.set_ydata(self.a_target_plot)

        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except:
            print("draw exception")

    @property
    def _state(self):

        Misc = np.array(
            [self.range / 1000.0, self.theta_missile, self.phi_missile,
             self.theta_los, self.phi_los, self.deri_range, self.d_theta_missile, self.d_phi_missile, self.d_theta_los, self.d_phi_los
             ])

        return np.concatenate([Misc.flat])

    def _store_dynamics(self, action):
        "keep tracking sequence "
        act = action[0]
        act2 = action[1]# [0][0]
        current = np.array(
            [self.time, self.theta_missile, self.phi_missile, self.theta_target, self.phi_target, self.theta_los, self.phi_los, self.range, self.d_theta_missile, self.d_phi_missile, self.d_theta_target, self.d_phi_target, self.d_theta_los, self.d_phi_los, self.azt, self.ayt, self.vm,
             self.vt, self.xm, self.ym, self.zm, self.xt, self.yt, self.zt, act, act2, self.deri_range, self.time_true])
        self.dynTrack.append(current)


    def _forward_dynamics(self, action):

        action[0] = max(action[0],-1)
        action[0] = min(action[0], 1)
        action[1] = max(action[1],-1)
        action[1] = min(action[1], 1)

        self.aym = action[0] * 20.0 * 9.8
        self.azm = action[1] * 20.0 * 9.8

        self.time_true = self.time_true + self.h

        self.ac1Now = action[0]
        self.ac2Now = action[1]

        self.time = self.time + 1

        self.ayt = 40.0 * np.sin(self.time_true + self.targetAcPhaseY)
        self.azt = 40.0 * np.sin(self.time_true + self.targetAcPhaseY)

        self.R_P = self.range

        self.RXYZ = np.sqrt((self.xt - self.xm) * (self.xt - self.xm) + (self.yt - self.ym) * (self.yt - self.ym) + (self.zt - self.zm) * (self.zt - self.zm)
        )

        self.rho = self.vt/self.vm
        self.d_theta_missile = self.azm / self.vm - self.d_phi_los * np.sin(self.theta_los) * np.sin(
            self.phi_missile) - self.d_theta_los * np.cos(self.phi_missile)

        self.d_phi_missile = self.aym / (self.vm * np.cos(self.theta_missile)) + self.d_phi_los * np.sin(self.theta_los) * np.cos(
            self.phi_missile) * np.tan(self.theta_missile) - self.d_theta_los * np.sin(self.phi_missile) * np.tan(
            self.theta_missile) - self.d_phi_los * np.cos(self.theta_los)
        self.d_theta_target = self.azt / (self.vt) - self.d_phi_los * np.sin(self.theta_los) * np.sin(
            self.phi_target) - self.d_theta_los * np.cos(self.phi_target)
        self.d_phi_target = self.ayt / (self.vt * np.cos(self.theta_target)) + self.d_phi_los * np.sin(self.theta_los) * np.cos(
            self.phi_target) * np.tan(self.theta_target) - self.d_theta_los * np.sin(self.phi_target) * np.tan(
            self.theta_target) - self.d_phi_los * np.cos(self.theta_los)

        self.theta_missile = self.theta_missile + self.d_theta_missile * self.h
        self.phi_missile = self.phi_missile + self.d_phi_missile * self.h
        self.theta_target = self.theta_target + self.d_theta_target * self.h
        self.phi_target = self.phi_target + self.d_phi_target * self.h

        self.d_theta_los = (self.vt * np.sin(self.theta_target) - self.vm * np.sin(self.theta_missile)) / self.range
        self.d_phi_los = (self.vt * np.cos(self.theta_target) * np.sin(self.phi_target) - self.vm * np.cos(self.theta_missile) * np.sin(
            self.phi_missile)) / (self.range * np.cos(self.theta_los))

        self.theta_los = self.theta_los + self.d_theta_los * self.h
        self.phi_los = self.phi_los + self.d_phi_los * self.h

        self.deri_range = self.vt * np.cos(self.theta_target) * np.cos(self.phi_target) - self.vm * np.cos(self.theta_missile) * np.cos(
            self.phi_missile)
        self.range = self.range + self.deri_range * self.h

        self.zm = self.zt - self.range * np.sin(self.theta_los)
        tempXY = abs(self.range * np.cos(self.theta_los))

        Tp = 288.16 - 0.0065 * self.zm
        if self.zm > 11000:
            Tp = 216.66

        Mach_Num = self.vm / (np.sqrt(1.4 * 288 * Tp))
        self.Mach_Num = Mach_Num

        CD0 = 0.02
        if Mach_Num > 0.93:
            CD0 = 0.02 + 0.2 * (Mach_Num - 0.93)
        if Mach_Num > 1.03:
            CD0 = 0.04 + 0.06 * (Mach_Num - 1.03)
        if Mach_Num > 1.10:
            CD0 = 0.0442 - 0.007 * (Mach_Num - 1.10)

        rho = 1.15579 - 1.058e-4 * self.zm + 3.725e-9 * self.zm * self.zm - 6e-14 * self.zm* self.zm* self.zm
        Q = 0.5 * rho * self.vm * self.vm
        K = 0.2
        if Mach_Num > 1.15:
            K = 0.2 + 0.246 * (Mach_Num - 1.15)

        m = 90.035
        T = 0

        if self.time_true < self.T_B:
            m = 113.205 - 3.31 * self.time_true
            T = 7500

        Di = (K * m * m * (self.aym * self.aym + self.azm * self.azm))/Q
        D0 = CD0 * Q
        D = D0 + Di
        self.D_vm = (T - D) / m - 9.8 * (
                np.cos(self.phi_missile) * np.cos(self.theta_missile) * np.sin(self.theta_los) + np.sin(self.theta_missile) * np.cos(
                self.theta_los))
        self.vm = self.vm + self.D_vm * self.h

        if abs(self.phi_los) < np.pi / 2:
            self.xm = self.xt - tempXY * np.cos(self.phi_los)
            self.ym = self.yt - tempXY * np.sin(self.phi_los)
        else:
            if self.phi_los > 0:
                self.xm = self.xt + tempXY * np.cos(np.pi - self.phi_los)
                self.ym = self.yt - tempXY * np.sin(np.pi - self.phi_los)
            else:
                self.xm = self.xt + tempXY * np.cos(np.pi - abs(self.phi_los))
                self.ym = self.yt + tempXY * np.sin(np.pi - abs(self.phi_los))



        Tran = np.array([[np.cos(self.theta_los) * np.cos(self.phi_los), np.sin(self.theta_los), -np.cos(self.theta_los) * np.sin(self.phi_los)],
                         [-np.sin(self.theta_los) * np.cos(self.phi_los), np.cos(self.theta_los), np.sin(self.theta_los) * np.sin(self.phi_los)],
                         [np.sin(self.phi_los), 0, np.cos(self.phi_los)]])

        VTLos = np.array([self.vt * np.cos(self.theta_target) * np.cos(self.phi_target),
                          self.vt * np.sin(self.theta_target),
                          - self.vt * np.cos(self.theta_target) * np.sin(self.phi_target)])

        DPT = np.dot(Tran.T, VTLos)


        self.d_xt = DPT[0]
        self.d_yt = DPT[1]
        self.d_zt = DPT[2]
        #self.xt = self.xt + self.d_xt * self.h
        #self.yt = self.yt + self.d_yt * self.h
        #self.zt = self.zt + self.d_zt * self.h
        if self.range >4:
            if self.time < self.target_store.shape[0] - 2:
                xyz_target = self.target_store[int(self.time + 1)]
            else:
                xyz_target = self.target_store[self.target_store.shape[0]-2]
            self.xt = xyz_target[0]
            self.yt = xyz_target[1]
            self.zt = xyz_target[2]

    def step(self, action):
        done = False
        temp_observation = []
        self.plot_iter = self.plot_iter + 1

        if self.range < 1:
            #print('last one meter')
            self.h = 0.00005
            self.num_subIter = 100

        i = 0
        while i < self.num_subIter:
            i = i + 1
            self._forward_dynamics(action)
            temp_observation.append(np.copy(self._state))
            if self.range < 0.1:
                self.h = 0.00001
                self.num_subIter = 500

            if self.range < 0.2:
                self.h = 0.0000001
                self.num_subIter = 50000
            if self.range < 0.002:
                self.h = 0.000000001
                self.num_subIter = 5000000
            if self.range < 1 and self.range > 0.03:
                print('Dist:{:.6f} theta_l:{:.5f}  phi_l:{:.5f}  D_theta_l:{:.5f} D_phi_l:{:.5f}'.format(
                    self.range, self.theta_los, self.phi_los, self.d_theta_los, self.d_phi_los))
                self.Terminal_LOS = np.array([self.range, self.theta_los, self.phi_los, self.d_theta_los, self.d_phi_los])
                self._store_dynamics(action)
            if self.R_P < self.range or self.range < 0:
                los_base = 'los_MD'
                np.save(los_base + str(self.file_num) + '.npy',self.Terminal_LOS)
                np.save('MD' + str(self.file_num) + '.npy', np.array([self.range, self.time_true]))
                run_base = 'run'
                np.save(run_base + str(self.file_num) + '.npy', np.array(self.dynTrack))
                done = True
                #input()
                break

        self._store_dynamics(action)
        print('Missile Velocity:{:.2f} Current Time:{:.4f}  Mach:{:.3f}'.format(self.vm, self.time_true, self.Mach_Num))
        next_observation = np.hstack((temp_observation[temp_observation.__len__() - 1]))#[0:5], mean_obs[5:11]))

        if self.range < 0:
            done = True
        if self.plot_iter % 5 == 0 and self.render:
            self._plotFig()

        return next_observation, done

if __name__ == '__main__':
    env = missile_env()
    env.reset_recursive()
    for i in range(1):
        env.reset_recursive()
        env.step(np.array([1,  1]))
        print('curr iter',i)
    state = env._state
    env.step(np.array([10,  10]))