import numpy as np
from matplotlib import pyplot as plt
import os
from utils.PSUtils.albedo import extract_albedo_from_image
from SRT3 import MPS_SCPS, MPS_SCPS_robust

from utils.PSUtils.render import render_Lambertian
from utils.PSUtils.eval import evalsurfaceNormal

def runplot(albedo, mode, startnum, endnum, seed):
    
    mae = []
    Data_folder = r'utils/PSUtils/sample/bunny'
    N_path = os.path.join(Data_folder, 'normal.npy')
    mask_path = os.path.join(Data_folder, 'mask.npy')
    for num_lights in range(startnum,endnum+1):
        print(num_lights)
       
        L_path = f'utils/PSUtils/sample/lights/True,0.5/sample_sphere_scaled_light_{num_lights}_3.npy' 
        L_dir = np.load(L_path)
        # print(L_dir.shape)
        mask = np.load(mask_path)
        # print(mask)
        N_gt = np.load(N_path)
        # print(N_gt.shape)

        h, w = mask.shape
        f = len(L_dir)
        if albedo=='sv':
            image_path = 'utils/PSUtils/svalbedo2.jpg'  
              
            albedo_map = extract_albedo_from_image(image_path, h,w,f,seed)

        elif albedo=='unif':
            albedo_map = np.ones([h, w, f])
            np.random.seed(seed)
            chrom = np.random.random(f)
            albedo_map = albedo_map * chrom[np.newaxis, np.newaxis, :]    
        
        img_set = render_Lambertian(N_gt, L_dir, mask, albedo_map, method_type='MPS', render_shadow=True)
        
        if mode=='rob':
            method_set = ['rob_Fact']
            for method in method_set:
                
                [N_est, reflectance] = MPS_SCPS_robust.run_MPS_SCPS_rob(img_set, mask, L_dir, method)
                
                [error_map, MAE, MedianE] = evalsurfaceNormal(N_gt, N_est, mask)
                # img_avg = error_map
                
                # img_avg = N_est
                # plt.imshow(img_avg, cmap='jet')
                # plt.title('Average of All Spectral Bands')
                # plt.show()
                print(method, MAE)
                mae.append(MAE)
                

        elif mode=='norm':
            method_set = ['Fact']
            for method in method_set:
                [N_est, reflectance] = MPS_SCPS.run_MPS_SCPS(img_set, mask, L_dir, method)
                
                [error_map, MAE, MedianE] = evalsurfaceNormal(N_gt, N_est, mask)
                # img_avg = error_map
                # #img_avg = N_est
                # plt.imshow(img_avg, cmap='jet')
                # plt.title('Average of All Spectral Bands')
                # plt.show()
                print(method, MAE)
                mae.append(MAE)  
                     
    
    return mae    


      




##### BLOCK 1 - to run graph of sv using norm   #######
# seed = 50 
# x = runplot('sv','norm',4,24,seed)
# print(x)
# lights = list(range(4, 25))
# plt.plot(lights, x, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for sv albedo with normal mode')
# plt.show()





###### BLOCK 2 - to run graph of unif using norm   #######
# seed = 50 
# x = runplot('unif','norm',4,24,seed)
# print(x)
# lights = list(range(4, 25))
# plt.plot(lights, x, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for uniform albedo with normal mode')
# plt.show()






###### BLOCK 3 - to run graph of unif using rob   #######
# num_iter = 100
# maes = np.zeros(8)
# for j in range(num_iter):
#   seed = 50 + j
#   x = runplot('unif','rob',17,24,seed)
#   maes = maes + np.array(x)    
# maes = maes/num_iter
# print(maes)
# lights = list(range(17, 25))
# plt.plot(lights, maes, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for uniform albedo with robust mode')
# plt.show()
  







###### BLOCK 4 - to run graph of sv using rob   #######
# num_iter = 100
# maes = np.zeros(8)
# for j in range(num_iter):
#   seed = 50 + j
#   x = runplot('sv','rob',17,24,seed)
#   maes = maes + np.array(x)    
# maes = maes/num_iter
# print(maes)
# lights = list(range(17, 25))
# plt.plot(lights, maes, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for sv albedo with robust mode')
# plt.show()




########### BLOCK 5 - to run graph of both albedos using norm #############
# seed = 50

# x_sv = runplot('sv', 'norm', 4, 24, seed)
# print(x_sv)

# x_unif = runplot('unif', 'norm', 4, 24, seed)
# print(x_unif)

# lights = list(range(4, 25))
# plt.plot(lights, x_unif, marker='o', linestyle='-', color='b', label='uniform albedo')
# plt.plot(lights, x_sv, marker='o', linestyle='-', color='r', label='sv albedo')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for sv and uniform albedo with normal mode')
# plt.legend()
# plt.show()





########### BLOCK 6 - to run graph of both albedos using rob #############
# num_iter = 100
# maes_unif = np.zeros(8)
# maes_sv = np.zeros(8)
# for j in range(num_iter):
#     seed = 50 + j
#     x_unif = runplot('unif', 'rob', 17, 24, seed)
#     maes_unif += np.array(x_unif)
# maes_unif = maes_unif / num_iter

# for j in range(num_iter):
#     seed = 50 + j
#     x_sv = runplot('sv', 'rob', 17, 24, seed)
#     maes_sv += np.array(x_sv)

# maes_sv = maes_sv / num_iter
# lights = list(range(17, 25))
# plt.plot(lights, maes_unif, marker='o', linestyle='-', color='b', label='uniform albedo')
# plt.plot(lights, maes_sv, marker='o', linestyle='-', color='r', label='sv albedo')
# plt.xlabel('Number of lights')
# plt.ylabel('MAE')
# plt.title('MAE vs num_lights for uniform and sv albedo with robust mode')
# plt.legend()
# plt.show()
