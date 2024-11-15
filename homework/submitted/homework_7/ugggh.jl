using Random, SpecialFunctions, Distributions;
using Plots;
using Optim;

# meshgrid function replacement for Julia
function meshgrid(x, y)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end;

# Define the sim_ps function to generate a point spread function (PSF) image
function sim_img(N, λ, NA, camera_scale, fine_Scale, Nₚ, center=(0, 0), B=0)
    block_size = camera_scale / fine_scale
    fine_N = N * block_size

    # define fine gride coordinates for x and y
    x = (-fine_N ÷ 2 : fine_grid_N ÷ 2 - 1) * fine_scale
    x = (-fine_N ÷ 2 : fine_grid_N ÷ 2 - 1) * fine_scale

    # generate 2D mesh grid for x and y coords
    X, Y = meshgrid(x, y)

    # extgract center coordinates
    xc, yc = center
    r = sqrt.((X .- xc).^2 .+ (Y .- yc).^2)

    v = (2π / λ) * NA * r

    # Initialize fine scale psf with zeros
    psf_fine = zeros(size(v))

    # define conditions based on v to account for v = 0
    for i = 1:1:length(v)
        if v[i] == 0
            psf_fine[i] = 1
        else
            psf_fine[i] = 4 * (besselj(1, v[i]) / v[i])^2
        end
    end

    # normalize
    psf_fine ./= sum(psf_fine)
    
    # initialize camera scale PSF
    psf_camera = zeros(N_camera, N_camera)
    
    # sum fine scale blocks into camera-scale pixels
    for i = 1:1:length(N_camera)
        for j = 1:1:length(N_camera)
            block = psf_fine[(i -1 ) * block_size + 1  :  i * block_size, (j-1) * block_size + 1  :  j * block_size]
            psf_camera[i, j] = sum(block)
        end
    end
    
    # Normalize and scale
    psf_camera ./= sum(psf_camera)
    psf_camera .*= N_photon
    
    # add poisson noise to the psf and the background
    noisy_psf = rand.(Poisson.(psf_camera))
    background = rand.(Poisson(B), N_camera, N_camera)
    
    # combine the noisy psf and the background
    final_image = noisy_psf .+ background

    return final_image
end;

# Define the calculate_MLE function to estimate parameters using Maximum Likelihood Estimation
function calc_MLE(intensities, initial_guess; camera_scale=0.01, origin_center=true)
    N = size(intensities, 1)
    x = collect(1:N)
    y = collect(1:N)
    
    if origin_center
        x .-= (N - 1) / 2
        y .-= (N - 1) / 2
    end
    
    X, Y = meshgrid(x, y)

    function mle_obj(params)
        xc, yc, A0, sigma, B = params
        model = A0 * exp.(-((X .- xc).^2 .+ (Y .- yc).^2) / (2 * sigma^2)) .+ B
        sum((model .- intensities) .* log.(max.(model, 1)))
    end

    results = optimize(mle_obj, Float64.(initial_guess))
    xc_est, yc_est, A0_est, sigma_est, B_est = Optim.minimizer(results)
    
    if camera_scale != 0
        xc_est *= camera_scale
        yc_est *= camera_scale
    end

    return xc_est, yc_est, A0_est, sigma_est, B_est
end;

# Define the calculate_rms_error function to calculate RMS error between calculated and true positions
function calculate_rms_error(calculated_positions, true_positions)
    if length(calculated_positions) == 2
        calculated_positions = hcat(calculated_positions...)
    end
    if length(true_positions) == 2
        true_positions = hcat(true_positions...)
    end
    
    squared_diffs = (calculated_positions .- true_positions).^2
    rms_error = sqrt(mean(sum(squared_diffs, dims=2)))
    
    return rms_error
end;

# Main simulation parameters
N_camera = 7
wavelength = 0.510
NA = 0.9
N_photons = 1000
camera_scale = 0.1
fine_scale = 0.01
bg = 10

m = 100
initial_guess = [0.0, 0.0, 100.0, 1.0, 0.0]

true_x_list, true_y_list = Float64[], Float64[];
x_error_list, y_error_list = Float64[], Float64[];


# Simulation loop
for i = 1:1:m
    center_x = rand(Uniform(-0.05, 0.05))
    center_y = rand(Uniform(-0.05, 0.05))
    current_center = (center_x, center_y)

    image = sim_ps(N_camera, wavelength, NA, camera_scale, fine_scale, N_photons, center=current_center, B=bg)
    
    MLE = calc_MLE(image, initial_guess; camera_scale=camera_scale, origin_center=true)
    
    x_MLE, y_MLE = MLE[1], MLE[2]
    x_error = x_MLE - current_center[1]
    y_error = y_MLE - current_center[2]

    push!(true_x_list, center_x)
    push!(x_error_list, x_error)

    push!(true_y_list, center_y)
    push!(y_error_list, y_error)
end

# Plotting
scatter(true_x_list, x_error_list, xlabel="True X-position (μm)", ylabel="∆X (μm)", legend=false)





###################################
# GRAVEYARD
###################################

# x = rand(Uniform(-0.05, 0.05))
# y = rand(Uniform(-0.05, 0.05))
# cent = (x, y)
# test_img = sim_ps(N_camera, wavelength, NA, camera_scale, fine_scale, N_photons, center=cent, B=bg)

# heatmap(test_img)

# MLE = calculate_MLE(test_img, initial_guess; camera_scale=camera_scale, origin_center=true)
# x_MLE, y_MLE = MLE[1], MLE[2]
# Δx = x_MLE - cent[1]
# Δy = y_MLE - cent[2]

# rms_error = sqrt(Δx^2 + Δy^2)