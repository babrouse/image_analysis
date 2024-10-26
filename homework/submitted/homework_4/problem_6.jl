# Image Analysis Homework 4 - Problem 6
# Bret A. Brouse Jr. - 10.25.2024

# Preload packages
using Images;
using Plots;
using SpecialFunctions, Random, Distributions;


# Functions, first two are from last homework
# euclidean distance calculator
function euclid_dist(cent, scale, a, b)
    A = (a - cent) * scale
    B = (b - cent) * scale
    C = sqrt(A^2 + B^2)

    return C
end;

function psf(N::Int, λ::Float64, aper::Float64, scale::Float64)
    # Make a grid of distances from the center of an NxN matrix
    center = (N - 1) / 2 # calc the center
    r = zeros(N, N) # initiate a matrix of distances from center

    # for distances from the center, calculated the euclidean distance
    for i = 1:1:N
        for j = 1:1:N
            r[i, j] = euclid_dist(center, scale, i, j)
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

# New function for pixelating our psf
function psf_pixelization(N::Int, 
                          λ::Float64, 
                          aper::Float64, 
                          cam_scale::Float64, 
                          fine_scale::Float64)
    # need to calculate the psf for a fine grid
    fine_N = Int(N * cam_scale / fine_scale)
    fine_psf = psf(fine_N, λ, aper, fine_scale)

    # down sample that feller
    coarse_psf = zeros(N, N) # initiate
    block_size = Int(cam_scale / fine_scale) # define the block size

    for i=1:1:N, j=1:1:N # iterate through using block sizes
        fine_i = ((i - 1) * block_size + 1):(i * block_size)
        fine_j = ((j - 1) * block_size + 1):(j * block_size)
        coarse_psf[i, j] = sum(fine_psf[fine_i, fine_j])
    end

    return coarse_psf / sum(coarse_psf) # return the normalized version
end;

function psf_pixelization_noisy(N::Int, 
    λ::Float64, 
    aper::Float64, 
    cam_scale::Float64, 
    fine_scale::Float64, 
    Nₚ::Int)

    # need to calculate the psf for a fine grid
    fine_N = Int(N * cam_scale / fine_scale)
    fine_psf = psf(fine_N, λ, aper, fine_scale)

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

    return noisy_psf # return the normalized version
end;

# variables to tweak
N = 15;
λ🟢 = 0.5;
aper₁ = 0.9;
fine_scale = 0.01;
cam_scale = 0.1;

pixel_psf = psf_pixelization(N, λ🟢, aper₁, cam_scale, fine_scale)
pixel_psf_map = heatmap(pixel_psf, title="Pixelated PSF with λ=0.5μm, NA=0.9")

noisy_psf_50_1 = psf_pixelization_noisy(N, λ🟢, aper₁, cam_scale, fine_scale, 50)
noisy_psf_50_2 = psf_pixelization_noisy(N, λ🟢, aper₁, cam_scale, fine_scale, 50)
noisy_psf_500_1 = psf_pixelization_noisy(N, λ🟢, aper₁, cam_scale, fine_scale, 500)
noisy_psf_500_2 = psf_pixelization_noisy(N, λ🟢, aper₁, cam_scale, fine_scale, 500)

noisy_psf_map_50_1 = heatmap(noisy_psf_50_1, title="Noisy PSF with λ=0.5μm, NA=0.9, N_photon=50")
noisy_psf_map_50_2 = heatmap(noisy_psf_50_2, title="Noisy PSF with λ=0.5μm, NA=0.9, N_photon=50")
noisy_psf_map_500_1 = heatmap(noisy_psf_500_1, title="Noisy PSF with λ=0.5μm, NA=0.9, N_photon=500")
noisy_psf_map_500_2 = heatmap(noisy_psf_500_2, title="Noisy PSF with λ=0.5μm, NA=0.9, N_photon=500")

savefig(pixel_psf_map, "images/pixel_psf_1.png")
savefig(noisy_psf_map_50_1, "images/noisy_psf_50_1.png")
savefig(noisy_psf_map_50_2, "images/noisy_psf_50_2.png")
savefig(noisy_psf_map_500_1, "images/noisy_psf_500_1.png")
savefig(noisy_psf_map_500_2, "images/noisy_psf_500_2.png")