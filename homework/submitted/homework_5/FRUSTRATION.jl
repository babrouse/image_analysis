################
# Because I've fucked everything else up, time to work on this again
# from the goddamn start
################

################
# Prob 2, HW4
################


# Preload packagees
using SpecialFunctions, Random, Distributions;
using Plots;

function calc_psf(N, λ, NA, pixel_scale, xc::Float64, yc::Float64)
    # define a center
    cent = (N + 1) / 2

    x = collect(1:N) .- cent
    y = collect(1:N) .- cent

    xv = [pixel_scale * (i - xc / pixel_scale) for i in x] 
    yv = [pixel_scale * (j - yc / pixel_scale) for j in y]

    # create radial distances
    r = [sqrt(x^2 + y^2) for x in xv, y in yv]

    # calculate v
    v = (2π / λ) .* NA .* r

    # calculate the damn psf and account for v==0
    psf = [v == 0 ? 1.0 : 4 * (besselj(1, v) / v)^2 for v in v]

    return psf
end;

# parameters
psf1 = calc_psf(100, 0.5, 0.9, 0.01)
psf2 = calc_psf(100, 0.4, 0.9, 0.01)
psf3 = calc_psf(100, 0.4, 0.5, 0.01)

heatmap(psf3)



################
# Prob 6a, HW4
################
function pixelate_psf(N::Int, 
    λ::Float64, 
    aper::Float64, 
    cam_scale::Float64, 
    fine_scale::Float64,
    xc::Float64,
    yc::Float64)

    # need to calculate the psf for a fine grid
    fine_N = Int(N * cam_scale / fine_scale)
    fine_psf = calc_psf(fine_N, λ, aper, fine_scale, xc, yc)

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

# test parameters
N = 15;                 # camera pixels
λ = 0.5;                # wavelength in μm
NA = 0.9;               # numerical aperture
camera_scale = 0.1;     # camera scale (μm / px)
fine_scale = 0.01;      # fine grid scale (μm / px)
Nₚ = 500;               # number of photons
bg = 10;                # bg noise factor
xc, yc = 0.1, 0.1;      # offset in μm

coarse_psf = pixelate_psf(N, λ, NA, camera_scale, fine_scale, xc, yc)

heatmap(coarse_psf, color=:inferno)


################
# Prob 6b, HW4
################

function add_noise(psf_array, Nₚ)
    # scale the psf so that the total intensity = Nₚ
    scaled_psf = psf_array * Nₚ

    # add noise
    noisy_psf = [rand(Poisson(intensity)) for intensity in scaled_psf]

    return noisy_psf
end;

noisy_psf = add_noise(coarse_psf, Nₚ)
heatmap(noisy_psf, color=:inferno)




################
# Prob 7, HW4
################
# added xc and yc to previous calc_psf function
# Needed to incorporate more arguments for both the psf and adding noise
# Added B as a background noise parameter



################
# Prob 8, HW4
################

function bg_noise(psf_array, N, bg)
    noise = [rand(Poisson(bg)) for _ in 1:1:N, _ in 1:1:N]

    final_image = [psf_array[i, j] + noise[i, j] for i in 1:1:N, j in 1:1:N]

    return final_image
end;


function psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
    coarse_psf = pixelate_psf(N, λ, NA, camera_scale, fine_scale, xc, yc)
    noisy_psf = add_noise(coarse_psf, Nₚ)
    final_psf = bg_noise(noisy_psf, N, bg)

    return final_psf
end;



# test parameters
N = 15;                 # camera pixels
λ = 0.5;                # wavelength in μm
NA = 0.9;               # numerical aperture
camera_scale = 0.1;     # camera scale (μm / px)
fine_scale = 0.01;      # fine grid scale (μm / px)
Nₚ = 500;               # number of photons
bg = 10;                # bg noise factor
xc, yc = 0.1, 0.1;      # offset in μm

final_psf = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
heatmap(final_psf, color=:inferno)