import numpy as np


#========================================================
#
# cost functions:
#
def missile_costfn(state, action, deltas,theta_l, phi_l, horizon):


    part1 = np.abs(
        deltas[:, 6] + 4.0 * (deltas[:, 4] - 0.6 * (0.800 - phi_l) / 200.0))
    part2 = np.abs(
        deltas[:, 5] + 4.0 * (deltas[:, 3] - 0.5 * (-0.600 - theta_l) / 200.0))

    if horizon > 1:
        score = 1.0 * part1 + 2.0 * part2
    else:
        score = 1.0 * part1 + 1.0 * part2

    return score

