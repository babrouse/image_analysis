# Image Analysis Homework 4 - Problem 8
# Bret A. Brouse Jr. - 10.25.2024

# Preload packages
using Images;
using Plots;
using SpecialFunctions, Random, Distributions;


# Functions, first two are from last homework
# euclidean distance calculator
function euclid_dist(cent, scale, a, b, xc, yc)
    A = (a - cent) * scale - xc
    B = (b - cent) * scale - yc
    C = sqrt(A^2 + B^2)

    return C
end;

function psf(N::Int, λ::Float64, aper::Float64, scale::Float64, xc, yc)
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
function psf_pixelization_noisy(N::Int, 
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

# variables to tweak
N = 15;
λ🟢 = 0.5;
aper₁ = 0.9;
fine_scale = 0.01;
cam_scale = 0.1;
xc = 0.2;
yc = 0.3;
bg = 2.0;

noisy_psf_50_1 = psf_pixelization_noisy(N, λ🟢, aper₁, cam_scale, fine_scale, 50, xc, yc, bg)

noisy_psf_map_50_1 = heatmap(noisy_psf_50_1, title="λ=0.5μm, NA=0.9, N_photon=50, xc=3, yc=3")