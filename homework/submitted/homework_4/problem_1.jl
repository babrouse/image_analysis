# Image Analysis Homework 4 - Problem 1
# Bret A. Brouse Jr. - 10.23.2024

# I decided to go back to using Julia as I was spending 
# too much time fighting with Python syntax (because I'm just
# used to Julia at this point!)

# Preload packages
using Images, ImageMagick, FFTW, Plots

# functions
function fast_four(img) # movie reference
    four = fftshift(fft(img))
    amp = log.(abs.(four) .+ 1)
    
    return amp
end


# load the dang images
buster_ori = load("images/Buster_Keaton_General_Train_512.png")
buster_sin = load("images/Buster_Keaton_General_Train_512_sineMod.png")

# convert images to grayscale arrays
ori_array = channelview(Gray.(buster_ori));
sin_array = channelview(Gray.(buster_sin));

# compute the amplitudes using the fourier transform then visualize them
amp_ori = fast_four(ori_array);
amp_sin = fast_four(sin_array);

ori_plt = heatmap(amp_ori, 
                  title="amplitude (original)", 
                  color=:grays, 
                  axis=false, 
                  framestyle=:box
                  );

sin_plt = heatmap(amp_sin, 
                  title="amplitude (sine shift)", 
                  color=:grays, 
                  axis=false
                  );

plot(ori_plt, sin_plt, layout=(2, 1), size=(600, 800))