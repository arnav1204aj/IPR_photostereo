import numpy as np
from utils.PSUtils.pre_processing import  get_valid_mask
import tqdm
from tqdm import tqdm

import math

from SRT3.MPS_SCPS import solveFact




def irls_weights(residuals, sigma=5 * math.pi / 180):
    """Compute IRLS weights based on residuals with gentle adjustment."""
    return sigma ** 2 / (residuals ** 2 + sigma ** 2)

def solveFact_robust(img_set, mask, L, t_low=0.6, shadow_ratio=0.25, remove_ratio=0.25, max_iter=1000, tol=1e-4, sigma=5 * math.pi / 180):
    [H, W, f_org] = img_set.shape  
    m = img_set[mask].T  
    f, p = m.shape  

    thres_mask = get_valid_mask(m, t_low, shadow_ratio)  
    if f_org - f_org * remove_ratio > 3:
        mask_t_idx = np.argsort(-np.sum(thres_mask, axis=1))[:int(f_org - f_org * remove_ratio)]
        mask_t = np.zeros(f_org, dtype=bool)
        mask_t[mask_t_idx] = True
    else:
        mask_t = np.ones(f_org, dtype=bool)

    
    Normal, reflectance = solveFact(img_set[:, :, mask_t], mask, L[mask_t])

    for iteration in tqdm(range(max_iter), desc="IRLS iterations for solveFact_robust"):
        
        LN = L[mask_t] @ Normal[mask].T  

        
        residuals = np.linalg.norm(m[mask_t] - LN, axis=0)  

        
        weights = irls_weights(residuals, sigma=sigma)  
        weights = weights[np.newaxis, :]  

        
        weighted_m = m[mask_t] * weights  

        
        try:
            S_hat = np.linalg.lstsq(L[mask_t], weighted_m.T, rcond=None)[0].T  
        except np.linalg.LinAlgError:
            
            continue  
        
        N = S_hat / (np.linalg.norm(S_hat, axis=0, keepdims=True) + 1e-8)  

        
        iter_norm = np.linalg.norm(Normal[mask] - N.T)
        if iter_norm < tol:
            break
        Normal[mask] = N.T  

    reflectance_full = np.zeros((H, W, f_org))
    reflectance_full[mask] = img_set[mask] / (Normal[mask] @ L.T + 1e-8)  

    return [Normal, reflectance_full]





def run_MPS_SCPS_rob(img_set, mask, L, method='linear', max_iter=1000, tol=1e-4, remove_ratio=0.7):
   
    if method == 'rob_Fact':
        normal, reflectance = solveFact_robust(img_set, mask, L, max_iter, tol, remove_ratio)
    
    else:
        raise Exception('Unknown method name')

    return [normal, reflectance]
