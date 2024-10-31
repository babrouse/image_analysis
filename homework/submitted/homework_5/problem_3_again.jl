# Image Analysis Homework 5 - Problem 3
# Bret A. Brouse Jr. - 10.31.2024

################### NOTE #####################
# Basically got lost in the sauce for this problem and ended up redoing all of HW4 problems 3, 6, 7, and 8
# I think this should work a little better now.

# Preload packages
using Plots;
using SpecialFunctions, Random, Distributions;

# A bunch of functions from previous work
# the main psf calculation that allows for offset
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

# A function that pixelates the psf based on fine scaling
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

# add noise to the psf itself
function add_noise(psf_array, Nₚ)
    # scale the psf so that the total intensity = Nₚ
    scaled_psf = psf_array * Nₚ

    # add noise
    noisy_psf = [rand(Poisson(intensity)) for intensity in scaled_psf]

    return noisy_psf
end;

# add background noise
function bg_noise(psf_array, N, bg)
    noise = [rand(Poisson(bg)) for _ in 1:1:N, _ in 1:1:N]

    final_image = [psf_array[i, j] + noise[i, j] for i in 1:1:N, j in 1:1:N]

    return final_image
end;

# Put all the previous functions together and return a noisy psf able to be offset and add bg noise
function psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
    coarse_psf = pixelate_psf(N, λ, NA, camera_scale, fine_scale, xc, yc)
    noisy_psf = add_noise(coarse_psf, Nₚ)
    final_psf = bg_noise(noisy_psf, N, bg)

    return final_psf
end;


# parameters
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



##########################
# Problem 2a, HW5
n = 1000    # number of positions
x = collect(0:n-1);
I = 0.2 .* sqrt.(x)
xc = sum(x .* I) / sum(I)

# Problem 2b, HW5
w = rand.(Poisson(10), n)   # generate Poisson noise with mean 10
I_noise = 30 ./ (x .+ 20) .+ 0.05 .* w  # Intensity with noise
xc_noise = sum(x .* I_noise) / sum(I_noise)

############################
# Problem 3a, HW5
function calc_centroid(img)
    # Extract image dimensions
    N, M = size(img)

    # create coord grid for x and y
    x, y = 1:N, 1:M
    xv = repeat(x', M, 1)  # transpose to match dimensions
    yv = repeat(y, 1, N)

    # calculate weighted sum for centroid
    I_total = sum(img)
    xc = sum(xv .* img) / I_total
    yc = sum(yv .* img) / I_total

    return xc, yc
end;

# NEW PARAMETERS
N = 7;                  # camera pixels
λ = 0.510;              # wavelength in μm
NA = 0.9;               # numerical aperture
camera_scale = 0.1;     # camera scale (μm / px)
fine_scale = 0.01;      # fine grid scale (μm / px)
Nₚ = 500;               # number of photons
bg = 10;                # bg noise factor
xc, yc = 0.0, 0.0;      # offset in μm
M = 100;                # number of images to generate

centroids = []

# generate images and calculate centroids
for _ in 1:M
    img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
    x_cent, y_cent = calc_centroid(img)
    push!(centroids, (x_cent, y_cent))
end

# convert centroid list to arrays for RMS calc
x_cent_values = [c[1] for c in centroids]
y_cent_values = [c[2] for c in centroids]

# calculate RMS error based on center
x0, y0 = (N + 1) / 2, (N + 1) / 2 # Expected center
rms_error = sqrt(mean((x_cent_values .- x0).^2 + (y_cent_values .- y0).^2))

println("RMS Error of centroid localization: ", rms_error)

# plot this histogram
rms_histo = histogram(x_cent_values, 
                    bins=37, 
                    title="calculated x_centroid (1000 images)", 
                    xlabel="x position", 
                    ylabel="frequency", 
                    legend=false, 
                    color=:purple,
                    size=(800,600))

# RMS error: 0.0742


############################
# Problem 3b, HW5
# Generate a range of Nₚ values
Nₚ_values = exp10.(range(log10(40), log10(40000), length=10))
println(Nₚ_values)

# NEW PARAMETERS
N = 7;                  # camera pixels
λ = 0.510;              # wavelength in μm
NA = 0.9;               # numerical aperture
camera_scale = 0.1;     # camera scale (μm / px)
fine_scale = 0.01;      # fine grid scale (μm / px)
Nₚ = 500;               # number of photons
bg = 10;                # bg noise factor
xc, yc = 0.0, 0.0;      # offset in μm
M = 100;                # number of images to generate

# array to store rms errors
rms_errors = [];

# Now loop through the new Nₚ values
for Nₚ in Nₚ_values
    centroids = []

    # simulate M imagees for current values (some copy past from above)
    for _ in 1:M
        img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
        x_cent, y_cent = calc_centroid(img)
        push!(centroids, (x_cent, y_cent))
    end
    
    x_cent_values = [c[1] for c in centroids]
    y_cent_values = [c[2] for c in centroids]
    
    x0, y0 = (N + 1) / 2, (N + 1) / 2 # Expected center
    rms_error = sqrt(mean((x_cent_values .- x0).^2 + (y_cent_values .- y0).^2))

    push!(rms_errors, rms_error)
end;

# plot those bad jessies
rms_photon_plt = plot(Nₚ_values, rms_errors, 
                      xscale=:log10, yscale=:log10, 
                      xlabel="Nₚ", ylabel="RMS error (pixels)", 
                      title="RMS error vs Nₚ", legend=false, size=(800, 600))


                      

############################
# Problem 3c, HW5
# New parameter
Nₚ_values = [1000, 100000];

true_position = [(0.0, 0.0), (0.3, 0.3), (-0.3, 0.0)];

Δx_dict = Dict{Tuple{Float64, Float64}, Vector{Float64}}()

for Nₚ in Nₚ_values
    for (x0, y0) in true_position
        Δx = []

        # generate images and calculate centroids
        for _ in 1:M
            img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
            x_cent, y_cent = calc_centroid(img)
            
            # calculate delta for the image instead this time
            push!(Δx, x_cent - (N + 1) / 2 + x0) # adjust for offset x0
        end

        # store these values
        Δx_dict[(Nₚ, x0)] = Δx
    end
end

# plot the histogram
for (Nₚ, x0) in keys(Δx_dict)
    Δx = Δx_dict[(Nₚ, x0)]
    hist = histogram(Δx, bins=50, 
                     title="Δx histo (Nₚ=$Nₚ, x0=$x0)", 
                    xlabel="Δx (pixels)", ylabel="frequency", legend=false, 
                    xlims=[-0.4, 0.4], size=(800, 600), linewidth=0, color=:purple)
    display(hist)
end



############################
# Problem 3d, HW5

# parameter changee
Nₚ = 1000;               # number of photons
M = 100;                # number of images to generate

# define true positions (p, p) from -0.5 to 0.5 px
p_values = collect(-0.5:0.1:0.5);

mean_Δx_no_noise, mean_Δx_noisy = [], [];

# loop over the p values for noise and no noise conditions
for p in p_values
    Δx_no_noise, Δx_noisy = [], []

    # sim images again and againa nd angaifdnalfkjhdaf
    for _ in 1:M
        # first without noise
        img_no_noise = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, 0.0)
        x_cent, y_cent = calc_centroid(img_no_noise)
        push!(Δx_no_noise, x_cent - (N + 1) / 2 + p)

        
        # now with noise
        img_noisy = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, 100.0)
        x_cent, y_cent = calc_centroid(img_noisy)
        push!(Δx_noisy, x_cent - (N + 1) / 2 + p)
    end

    # calc mean Δx for both
    push!(mean_Δx_no_noise, mean(Δx_no_noise))
    push!(mean_Δx_noisy, mean(Δx_noisy))
end

noise_rel_plots = plot(p_values, mean_Δx_no_noise, label="no noise", xlabel="True position p (pixels)", ylabel="Mean Δx (pixels)", size=(800, 600))
plot!(p_values, mean_Δx_noisy, label="with noise (B=100)")



                      
#= EVERYTHING WORKS UP TO HERE =#
############################
# Problem 1, HW5


