

### Tests for new parameter class

import numpy as np

import parameters
import word

def two_exp_parameter_tests():

    times = np.arange(1000)/1000.0

    ### single word parameters
    t0 = 0.1
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 5.0

    ### parameter object
    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=log_amp, skew=log_skew, log=True)

    print("Does TwoExpParameters store parameters correctly?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.t0))
    print("peak time log_scale = -4 : \t ... \t " + str(theta.log_scale))
    print("peak time log_amp = 5 : \t ... \t " + str(theta.log_amp))
    print("peak time log_skew = 2 : \t ... \t " + str(theta.log_skew) + "\n")

    print("Does TwoExpParameters convert log-parameters correctly?")
    print("peak time scale = 0.018 : \t ... \t " + str(theta.scale))
    print("peak time amp = 148.4 : \t ... \t " + str(theta.amp))
    print("peak time skew = 7.38 : \t ... \t " + str(theta.skew) + "\n")

    ### Combined parameter object with two words, scale and skew individually defined
    t0_2 = 0.5
    log_scale_2 = -6.0
    log_skew_2 = -3.0
    log_amp_2 = 3.0

    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, log_skew, t0_2, log_scale_2, log_amp_2, log_skew_2], 2, log=True)

    print("Does TwoExpCombined store parameters correctly?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("peak time log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("peak time log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("peak time log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("peak time log_scale_2 = -6 : \t ... \t " + str(theta.all[1].log_scale))
    print("peak time log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("peak time log_skew_2 = -3 : \t ... \t " + str(theta.all[1].log_skew) + "\n")


    ### test with scale_locked only:
    theta = parameters.TwoExpCombined([t0, log_amp, log_skew, t0_2, log_amp_2, log_skew_2, log_scale], 2, scale_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when scale_locked=True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("peak time log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("peak time log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("peak time log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("peak time log_scale_2 = -4 : \t ... \t " + str(theta.all[1].log_scale))
    print("peak time log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("peak time log_skew_2 = -3 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ### test with skew_locked only:
    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, t0_2, log_scale_2, log_amp_2, log_skew], 2, skew_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when skew_locked=True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("peak time log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("peak time log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("peak time log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("peak time log_scale_2 = -6 : \t ... \t " + str(theta.all[1].log_scale))
    print("peak time log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("peak time log_skew_2 = 2 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ### test with both scale_locked and skew_locked:
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_skew, log_scale], 2, scale_locked=True, skew_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when scale_locked=True and skew_locked = True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("peak time log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("peak time log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("peak time log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("peak time log_scale_2 = -4 : \t ... \t " + str(theta.all[1].log_scale))
    print("peak time log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("peak time log_skew_2 = 2 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ###

    ###

    return



def test_word():

    return

def main():

    two_exp_parameter_tests()

    return

if __name__ == "__main__":

    main()

