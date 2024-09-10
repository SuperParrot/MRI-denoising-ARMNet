import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time

from interface import Interface

if __name__=='__main__':
    interface=Interface()

    '''
    model_idxes = list(range(5, 80, 5))
    model_idxes.append(79)

    for model_idx in model_idxes:
        print('Model %d:' % model_idx)

        noise_level_list = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20]
        for noise_level in noise_level_list:
            mae_val, psnr_val, ssim_val = interface.predict('./eval_list.txt', noise_level=noise_level,
                                                            assigned_modelNames=['params_' + str(model_idx) + '.pkl'])
            print('Noise level:%.2lf, MAE: %.6lf\tPSNR: %.6lf\tSSIM: %.6lf' % (noise_level, mae_val, psnr_val, ssim_val))
    '''

    #interface.predict('./manual_list.txt', noise_level=0.05)
    noise_level_list = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20]
    for noise_level_idx in range(len(noise_level_list)):
        noise_level = noise_level_list[noise_level_idx]
        mae_val, psnr_val, ssim_val, lpips_val = interface.predict('./eval_list.txt', noise_level=noise_level)
        print('Noise level:%.2lf, MAE: %.6lf\tPSNR: %.6lf\tSSIM: %.6lf\tLPIPS: %.6lf' % (noise_level, mae_val, psnr_val, ssim_val, lpips_val))

    #mae_val, psnr_val, ssim_val = interface.predict('./test_list.txt', noise_level=0.08)
