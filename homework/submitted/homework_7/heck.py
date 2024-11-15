import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from skimage import io
from scipy.optimize import minimize
from scipy.ndimage import convolve
from skimage import restoration
from radialcenter_ImAnClass import radialcenter
import time

# image simulation with noise
def sim_img(N, wavelength, NA, camera_scale, fine_scale, N_p, center=None, B=None):
    block_size = int(camera_scale / fine_scale)
    fine_N = N * block_size
    
    x = np.linspace(-fine_N // 2, fine_N // 2, fine_N) * fine_scale
    y = np.linspace(-fine_N // 2, fine_N // 2, fine_N) * fine_scale
    X, Y = np.meshgrid(x, y)
    
    if center:
        xc, yc = center
    else:
        xc, yc, = 0, 0
    
    r = np.sqrt((X - xc)**2 + (Y - yc)**2)
    v = (2 * np.pi / wavelength) * NA * r
    
    psf_fine = np.zeros_like(v)
    psf_fine[v == 0] = 1
    psf_fine[v != 0] = 4 * (j1(v[v != 0]) / v[v != 0])**2
        
        
    psf_fine /= psf_fine.sum()
    
    psf_camera = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            block = psf_fine[i * block_size : (i + 1) * block_size, 
                             j * block_size : (j + 1) * block_size]
            psf_camera[i, j] = np.sum(block)
            
    psf_camera /= np.sum(psf_camera)
    psf_camera *= N_p
    
    noisy_psf = np.random.poisson(psf_camera)
    
    bg = np.random.poisson(B, (N, N))
    
    final_img = noisy_psf + bg
        
    return final_img

# Maximum likelihood estimation function
def calc_MLE(img, initial_guess, camera_scale=0.1, orig_cent=True):
    N = img.shape[0]
    x = np.arange(N)
    y = np.arange(N)
    
    mid = (N - 1) / 2
    
    if orig_cent == True:
        x = x - mid
        y = y - mid
        
    X, Y = np.meshgrid(x, y)
    
    # define an mle objective function based on a gaussian
    def mle_obj(params, X, Y, data):
        xc, yc, A_0, sigma, B = params
        # gaussian model
        model = A_0 * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * sigma**2)) + B
        return np.sum(model - data * np.log(np.where(model > 0, model, 1)))
    
    results = minimize(mle_obj, initial_guess, args=(X, Y, img))
    x_est, y_est, A_0_est, sigma_est, B_est = results.x
    
    if camera_scale:
        x_est *= camera_scale
        y_est *= camera_scale
        
    return x_est, y_est, A_0_est, sigma_est, B_est
    
# calculating RMS error
def calc_rmse(est_list, true_list):
    
    est_list = np.array(est_list)
    true_list = np.array(true_list)
    
    if est_list.ndim == 1:
        est_list = est_list.reshape(1, -1)
    
    if true_list.ndim == 1:
        true_list = true_list.reshsape(1, -1)
        
    diffs = est_list - true_list
    sq_diffs = diffs**2
    rmse = np.sqrt(np.mean(np.sum(sq_diffs, axis=1)))
    
    return rmse

# Need to rewrite a gaussian psf function from homework 2 I think)
def gauss_psf(N, sigma):
    x = np.linspace(-N // 2, N // 2, N)
    y = np.linspace(-N // 2, N // 2, N)
    
    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2 + Y**2)
    
    # make the psf based on a gaussian
    psf = np.exp(-(r**2) / (2 * sigma**2))
    
    # normalize
    psf = psf / psf.sum()
    
    return psf


def calc_img_rmse(img_1, img_2):
    difference = img_1 - img_2
    squared_diff = np.square(difference)
    mean_squared_diff = np.mean(squared_diff)
    rms_error = np.sqrt(mean_squared_diff)
    return rms_error


# Parameters

N = 7
wavelength = 0.510
NA = 0.9
N_p = 1000
camera_scale = 0.1
fine_scale = 0.01
bg = 10

m = 100
guess = [0, 0, 100, 1, 0]


# #################################
# # 1a
# #################################

# # # test that img sim function works
# # test_img = sim_img(N, wavelength, NA, camera_scale, fine_scale, N_p, B=bg)
# # plt.imshow(test_img, cmap='viridis', aspect='auto')



# # additional parameters
# true_xs, error_xs = [], []
# true_ys, error_ys = [], []


# for _ in range(m):
#     xc, yc = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
    
#     cent = (xc, yc)
    
#     img = sim_img(N, wavelength, NA, camera_scale, fine_scale, N_p, center=cent, B=bg)
    
#     MLE = calc_MLE(img, guess, camera_scale, True)
    
#     x_MLE, y_MLE = MLE[0], MLE[1]
    
#     delta_x = x_MLE - cent[0]
#     delta_y = y_MLE - cent[1]
    
#     true_xs.append(xc)
#     true_ys.append(yc)
    
#     error_xs.append(delta_x)
#     error_ys.append(delta_y)
    
# plt.scatter(true_xs, error_xs)
# plt.xlabel('true x (μm)')
# plt.ylabel('∆X (μm)')


#################################
# 1b
#################################

# N_ps = np.logspace(np.log10(40), np.log10(40000), num=10)
# rmss, vals = [], []

# sigma = wavelength / (2 * NA)

# for p in N_ps:
#     true_list, est_list = [], []
    
#     for _ in range(m):
#         xc, yc = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
#         cent = (xc, yc)
        
#         img = sim_img(N, wavelength, NA, camera_scale, fine_scale, p, center=cent, B=bg)
        
#         MLE = calc_MLE(img, guess, camera_scale, True)
        
#         true_list.append(cent)
#         est_list.append((MLE[0], MLE[1]))
        
#     rms = calc_rmse(est_list, true_list)
#     rmss.append(rms)
    
#     val = sigma / np.sqrt(p)
#     vals.append(val)
    
# plt.loglog(N_ps, rmss, label=r'$RMS$')
# plt.loglog(N_ps, vals, label=r'$\frac{σ}{\sqrt{N_{photons}}}$')
# plt.xlabel(r'$N_{photons}$')
# plt.legend()


#################################
# 2a
#################################

# imgs = []

# for _ in range(m):
#     xc, yc = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
#     cent = (xc, yc)
    
#     img = sim_img(N, wavelength, NA, camera_scale, fine_scale, N_p, center=cent, B=bg)

#     imgs.append(img)
    
# imgs = np.array(imgs)

# start_time = time.time()
# for i in range(m):
#     xrs, yrs = radialcenter(imgs[i])

# end_time = time.time()

# elapsed = end_time - start_time
# print("total time taken: ", elapsed, " seconds.")



#################################
# 2b
#################################

# true_xs, error_xs = [], []
# true_ys, error_ys = [], []

# for _ in range(m):
#     xc, yc = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
#     cent = (xc, yc)
    
#     img = sim_img(N, wavelength, NA, camera_scale, fine_scale, N_p, center=cent, B=bg)
    
#     xrs, yrs = radialcenter(img)
    
#     delta_x = xrs - cent[0]
#     delta_y = yrs - cent[1]
    
#     true_xs.append(xc)
#     true_ys.append(yc)
    
#     error_xs.append(delta_x)
#     error_ys.append(delta_y)
    
# plt.scatter(true_xs, error_xs)
# plt.xlabel('true x (μm)')
# plt.ylabel('∆X (μm)')



#################################
# 2c
#################################

# N_ps = np.logspace(np.log10(40), np.log10(40000), num=10)
# rmss, vals = [], []

# sigma = wavelength / (2 * NA)

# for p in N_ps:
#     true_list, est_list = [], []
    
#     for _ in range(m):
#         xc, yc = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
#         cent = (xc, yc)
        
#         img = sim_img(N, wavelength, NA, camera_scale, fine_scale, p, center=cent, B=bg)
        
#         xrs, yrs = radialcenter(img)
        
#         true_list.append(cent)
#         est_list.append((xrs, yrs))
        
#     rms = calc_rmse(est_list, true_list)
#     rmss.append(rms)
    
#     val = sigma / np.sqrt(p)
#     vals.append(val)
    
# plt.loglog(N_ps, rmss, label=r'$RMS$')
# plt.loglog(N_ps, vals, label=r'$\frac{σ}{\sqrt{N_{photons}}}$')
# plt.xlabel(r'$N_{photons}$')
# plt.title("RMS error vs N_photons (radial center)")
# plt.legend()



#################################
# 3a
#################################
# gaussian psf function is written above with other functions

# 




#################################
# 3b
#################################
# N = 35
# sigma = 5

# img = io.imread("mouse_glial_cells_RBurdan_crop.png")
# io.imshow(img)

# psf = gauss_psf(N, sigma)
# gaussed_img = convolve(img, psf)
# noisy_gauss_img = np.clip(np.random.poisson(gaussed_img), 0, 255).astype(np.uint8)

# plt.imshow(noisy_gauss_img, cmap='gray')
# plt.title("noisy, blurry image!")

# plt.imshow(gaussed_img, cmap='gray')
# plt.title("image convolved with a gaussian (sigma=5, N=35)")



#################################
# 3c
#################################
# iters = 20

# decon_img = restoration.richardson_lucy(noisy_gauss_img, psf, iterations=iters, clip=False)

# plt.imshow(decon_img, cmap='gray')
# plt.title("deconvolved (20 iterations)")

# clip = 50

# clipped_decon_img = decon_img[clip:-clip, clip:-clip]
# clipped_decon_img = (255 * (clipped_decon_img - np.min(clipped_decon_img)) / 
#                             (np.max(clipped_decon_img) - np.min(clipped_decon_img))).astype(np.uint8)

# plt.imshow(clipped_decon_img, cmap='gray')

# clipped_img = img[clip:-clip, clip:-clip]

# rmse_decon_img = calc_img_rmse(img, decon_img)
# rmse_decon_img_clipped = calc_img_rmse(clipped_img, clipped_decon_img)


# print("deconvolved rmse prior to clipping: ", rmse_decon_img)
# print("deconvolved rmse after clipping: ", rmse_decon_img_clipped)



#################################
# 3d
#################################
# iter_list = np.arange(5, 205, 10)

# rmses, imgs = [], []

# clip = 50

# for iters in iter_list:
#     clipped_img = img[clip:-clip, clip:-clip]
    
#     decon_img = restoration.richardson_lucy(noisy_gauss_img, psf, iters, clip=False)
    
#     clipped_decon_img = decon_img[clip:-clip, clip:-clip]
#     clipped_decon_img = (255 * (clipped_decon_img - np.min(clipped_decon_img)) / 
#                                 (np.max(clipped_decon_img) - np.min(clipped_decon_img))).astype(np.uint8)
    
#     rmse = calc_img_rmse(clipped_img, clipped_decon_img)
    
#     rmses.append(rmse)
#     imgs.append(clipped_decon_img)
    
# plt.clf()
# plt.plot(iter_list, rmses)
# plt.xlabel("iterations")
# plt.ylabel("rms error")
    
    



#################################
# 3e
#################################
# min_rmse_img = imgs[rmses.index(min(rmses))]
# print(min_rmse_img)

# plt.imshow(min_rmse_img)
# plt.title("smallest rmse img")




























