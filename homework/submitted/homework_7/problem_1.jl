     #### Homework 7, Problem 1
     ## Bret A. Brouse Jr. - 11.13.2024

# Preload packages
using Random, Distributions, SpecialFunctions;
using Optim;
using Plots;

plotly()

# functions
# We want parameters x₀ and b to maximize likelihood of observing y
function mle_obj(params, x, y)
    x₀, b = params
    A = A_calc(x, x₀, b)

    # fine negative log likelihood
    log_like = sum(A .- y .* log.(A))

    return log_like
end;

function A_calc(x, x₀, b)
    A = 3 * abs.(x .- x₀).^b .+ 4
    
    return A
end;

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

# Centroid calc
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

function gauss_psf(x, y, A₀, xc, yc, σ, B)
    A = A₀ * exp(-((x - xc)^2 + (y - yc)^2) / (2 * σ^2)) + B

    return A
end;

function neg_log(params, img, x_grid, y_grid)
    A₀, xc, yc, σ, B = params
    
    # calculate expected intensities for each pixel
    exp_Is = []
    exp_Is = gauss_psf.(x_grid, y_grid, A₀, xc, yc, σ, B)
    # for i = 1:1:length(x_grid), j = 1:1:length(y_grid)
    #     push!(exp_Is, gauss_psf(x_grid[i], y_grid[j], A₀, xc, yc, σ, B))
    # end

    # negative log-likelihood assuming Poisson noise
    # log_likelihood = sum(exp_Is .- img .* log.(exp_Is .+ 1e-10))
    log_likelihood = sum(exp_Is .- img .* log.(max.(exp_Is, 1e-10) .+ 1e-10))


    return log_likelihood
end;

function mle_localization(img)
    # initial guess
    init_params = [maximum(img), size(img, 1) / 2, size(img, 2) / 2, 1.0, minimum(img)]
    #=  maximum(img) - guess for A₀ is maximum of the image
        sizes - reasonable to guess center
        1.0 - suppose that the standard deviation could be 1.0
        minimum(img) - guess for B which is background noise =#

    # create dang grids
    rows, cols = size(img)
    x_grid = repeat(collect(1:cols)', rows, 1)
    y_grid = repeat(collect(1:rows), 1, cols)

    # run optimization using Optim packagee
    result = optimize(params -> neg_log(params, img, x_grid, y_grid), init_params)

    # extract the optimized parameters
    A₀, xc, yc, σ, B = Optim.minimizer(result)

    return (xc, yc, A₀, σ, B)
end;

function rms_error(x₀, xc, y₀, yc)
    Δx = xc - x₀
    Δy = yc - y₀
    rms_err = sqrt(Δx^2 + Δy^2)

    return rms_err
end;

# New parameters
N = 7;
λ = 0.510;
NA = 0.9;
Nₚ = 1000;
camera_scale = 0.1;
fine_scale = 0.01;
bg = 10;
M = 100;

# rando numbers between -0.05 and 0.05
x_poss = rand(M) .* 0.1 .- 0.05;
y_poss = rand(M) .* 0.1 .- 0.05;

sim_imgs = [];
Δxs, Δys = [], [];
rms_errors = [];


#= generate an image using the psf function, extract the center using mle_local. function,
    push this to the delta x array for plotting =#
for i = 1:1:M
    img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, x_poss[i], y_poss[i], bg)
    xc, yc, _, _, _ = mle_localization(img)
    xc -= 3.5
    yc -= 3.5

    Δx = 0.01*(xc - x_poss[i])
    Δy = 0.01*(yc - y_poss[i])

    push!(sim_imgs, img)
    push!(Δxs, Δx)
    push!(Δys, Δy)
    push!(rms_errors, rms_error(x_poss[i], xc, y_poss[i], yc))
end;

plt_1a = scatter(x_poss, Δxs, 
                 xlabel="True x₀ (pixels)", 
                 ylabel="Δx", 
                 title="Δx vs true x₀ for MLE localization",
                 grid=true, 
                 legend=false, 
                 size=(800, 600)
)





########################
# GRAVEYARD
########################
test_img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, x_poss[1], y_poss[1], bg)
heatmap(test_img)
x_test, y_test, _, _, _ = mle_localization(test_img)
x_test -= 3.5
y_test -= 3.5

delta_x = 0.1*(x_test - x_poss[1])
delta_y = 0.1*(y_test - y_poss[1])