# Image Analysis Homework 5 - Problem 3
# Bret A. Brouse Jr. - 10.30.2024

# Preload packages
using Random, Statistics, Distributions, SpecialFunctions;
using Plots;

# functions
function centroid_calc(I_array, xᵢ)
    A = sum(I_array .* xᵢ) / sum(I_array)

    return A
end;

# euclidean distance calculator
function euclid_dist(cent, scale, a, b, xc, yc)
    A = (a - cent) * scale - xc
    B = (b - cent) * scale - yc
    C = sqrt(A^2 + B^2)

    return C
end;

function psf(N::Int, λ::Float64, aper::Float64, scale::Float64, xc=0.0, yc=0.0)
    # Make a grid of distances from the center of an NxN matrix
    center = (N - 1) / 2 # calc the center
    r = zeros(N, N) # initiate a matrix of distances from center

    # for distances from the center, calculated the euclidean distance
    for i = 1:1:N
        for j = 1:1:N
            r[i, j] = euclid_dist(center, scale, i, j, xc, yc)
        end
    end

    # now multiply the matrix r by some scalar values
    v = (2 * π / λ) * aper * r

    # initiate an array for PSF values
    psf_array = zeros(N, N)

    # Now do PSF calc for each point in psf_array
    for i = 1:1:N, j = 1:1:N
        if v[i, j] == 0
            psf_array[i, j] = 1 # special case for v = 0
        else
            # In the SpecialFunctions package, we get bessel functions
            psf_array[i, j] = 4 * (besselj1(v[i, j]) / v[i, j])^2 # given point spread function
        end
    end

    return psf_array / sum(psf_array)
end;

# Pixelization and noise
function img_sim(N::Int, 
    λ::Float64, 
    aper::Float64, 
    cam_scale::Float64, 
    fine_scale::Float64, 
    Nₚ::Int, 
    xc::Float64=0.0, 
    yc::Float64=0.0, 
    bg::Float64=0.0)

    # need to calculate the psf for a fine grid
    fine_N = Int(N * cam_scale / fine_scale)
    fine_psf = psf(fine_N, λ, aper, fine_scale, xc, yc)

    # down sample that feller
    coarse_psf = zeros(N, N) # initiate
    block_size = Int(cam_scale / fine_scale) # define the block size

    for i=1:1:N, j=1:1:N # iterate through using block sizes
    fine_i = ((i - 1) * block_size + 1):(i * block_size)
    fine_j = ((j - 1) * block_size + 1):(j * block_size)
    coarse_psf[i, j] = sum(fine_psf[fine_i, fine_j])
    end

    # scale and add noise for part (b)
    coarse_psf_scaled = coarse_psf * (Nₚ / sum(coarse_psf))
    noisy_psf = rand.(Poisson.(coarse_psf_scaled))

    # Add background noise
    bg_noise = rand.(Poisson(bg), N, N)
    noisy_psf = noisy_psf .+ bg_noise

    return noisy_psf # return the normalized version
end;

function cent_calc(I, orig=(0.0, 0.0))
    if length(size(I)) == 1
        n = length(I)
        x = 0:length(I)-1

        if orig ≠ (0.0, 0.0)
            center = div(n-1, 2)
            x = x .- center
        end
        x_cent = sum(x .* I) ./ sum(I)
        return (x_cent,)

    elseif length(size(I)) == 2
        m, n = size(I)
        x = 0:n-1
        y = 0:m-1

        if orig ≠ (0.0, 0.0)
            x_cent = (n - 1) / 2
            y_cent = (m - 1) / 2

            x .-= x_cent
            y .-= y_cent
        end

        x_wt = sum(I, dims=1)
        y_wt = sum(I, dims=2)

        x_cent = sum(x .* vec(x_wt)) / sum(I)
        y_cent = sum(y .* vec(y_wt)) / sum(I)

        return (x_cent, y_cent)
    end
end



# parameters
# In order: Number of images, coarse grid size, fine grid size
# scale, wavelength, numerical aperture, photon number, mean
M = 100;
N = 15;
fine_scale = 0.01;
scale = 0.1;
λ = 0.510;
aper = 0.9;
Nₚ = 50000;
bg = 10.0;
yc, xc = 0.0, 0.0;

sim_img = img_sim(N, λ, aper, scale, fine_scale, Nₚ, xc, yc, bg)
heatmap(sim_img, color=:inferno)

# I = vec(sim_img)
# cent_test = cent_calc(I)

cent_calc(sim_img)

centers, centroids, x_list = [], [], []

for i = 1:1:M
    img = img_sim(N, λ, aper, scale, fine_scalee, Nₚ, xc, yc, bg)
end

#= Tested the centroid calc multiple times with and without noise and it seems to be consistent.
    Basically I tried it with the image with things centered and get close to center for x and y
    components. When 1D, it returns roughly half the length of the array, as expected. Phew.
    It turns out with noise, this is really bad.
    =#





x=19;


#####################################
# Graveyard
#####################################
# N_camera = 7
# wavelength = 0.50
# NA = 0.9
# camera_scale = 0.1
# fine_scale = 0.01
# N_photons = 2000
# offset = (0.04, 0)
# bg = 2


# function img_check(img)
#     N = size(img, 1)    # assumes array is square
#     center_px = div(N - 1, 2)   # have to use integer division or get errors

#     rat_cent_r = img[center_px + 1, center_px +1] / img[center_px + 1, center_px+2]
#     println("Ratio 1: ", round(rat_cent_r, digits=2))
#     if abs(rat_cent_r - 1.06) < 0.18
#         println("pass")
#     else
#         println("failed")
#     end

#     rat_cent_l = img[center_px + 1, center_px +1] / img[center_px + 1, center_px]
#     println("Ratio 2: ", round(rat_cent_l, digits=2))
#     if abs(rat_cent_l - 1.81) < 0.4
#         println("pass")
#     else
#         println("failed")
#     end

#     rat_cent_u = img[center_px + 1, center_px +1] / img[center_px, center_px+1]
#     println("Ratio 1: ", round(rat_cent_u, digits=2))
#     if abs(rat_cent_u - 1.36) < 0.26
#         println("pass")
#     else
#         println("failed")
#     end
# end;