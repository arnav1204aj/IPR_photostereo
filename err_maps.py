import numpy as np
from matplotlib import pyplot as plt
import os
from utils.PSUtils.albedo import extract_albedo_from_image
from SRT3 import MPS_SCPS, MPS_SCPS_robust

from utils.PSUtils.render import render_Lambertian
from utils.PSUtils.eval import evalsurfaceNormal

def runmap(albedo, mode, num_lights,seed):
        
        Data_folder = r'utils/PSUtils/sample/bunny'
        N_path = os.path.join(Data_folder, 'normal.npy')
        mask_path = os.path.join(Data_folder, 'mask.npy')
    
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
            chrom = np.random.random(f)
            albedo_map = albedo_map * chrom[np.newaxis, np.newaxis, :] 
            # albedo_avg = albedo_map.mean(axis=2)
            # plt.imshow(albedo_avg, cmap='gray')
            # plt.title('Albedo Map (3D) - Average across bands')
            # plt.colorbar()
            # plt.show()   
        
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
                  
                     
    
        return MAE,error_map, N_est, img_set, albedo_map    

#alter these 3 params to get results
albedo = 'unif'       
mode = 'rob'
num_lights = 24




num_iter = 100
mae = 0
errmap = None
normals = None
rend_img = None
for i in range(num_iter):
 
 seed = 50 + i
 
 x,y,z,w,albedo_map = runmap(albedo,mode,num_lights,seed)
 print(x)
 mae = mae + x
 if i==0:
    errmap = y
    normals = z
    rend_img = w
 else:
    errmap = errmap + y
    normals = normals + z 
    rend_img = rend_img + w  

print(f'Net MAE = {mae/num_iter}')
error_map = errmap/num_iter
n_est = normals/num_iter
rendered_img = rend_img/num_iter
img_avg = error_map
                

plt.imshow(img_avg, cmap='jet')
plt.title('Error Map')
plt.show()

img_avg = n_est
                

plt.imshow(img_avg, cmap='brg')
plt.title('Normal Estimates')
plt.show()


albedo_avg = albedo_map.mean(axis=2)
plt.imshow(albedo_avg, cmap='gray')
plt.title('Albedo')
plt.colorbar()
plt.show() 



img_avg = rendered_img[:,:,0]
plt.imshow(img_avg, cmap='gray')
plt.title('Light 1 rendered image')
plt.show()

img_avg = rendered_img[:,:,1]
plt.imshow(img_avg, cmap='gray')
plt.title('Light 2 rendered image')
plt.show()

img_avg = rendered_img[:,:,23]
plt.imshow(img_avg, cmap='gray')
plt.title('Light 24 rendered image')
plt.show()


  