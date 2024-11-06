# Homework 6 Problem 1a
# Bret A. Brouse Jr. - 11.6.2024

# Preload packages
using Images, FileIO;
using Statistics;
using Plots;
gr()

# load tif as a 3D array and make them floats
img_stack = load("homework/submitted/homework_6/images/cilia_movie_crop.tif");
img_stack = Float64.(img_stack);
# this ends up being upside down so I mirror it vertically
img_stack = img_stack[end:-1:1, :, :]

size(img_stack)
# 196x396 resolution with 126 frames

# check to make sure a frame is in fact what it should look like
heatmap(img_stack[:, :, 37], color=:inferno, title="frame 37", size=(800, 600))

# Calculate median across temporal dimension
med_img = median(img_stack, dims=3)
heatmap(med_img[:, :, 1], color=:viridis, title="median img")
# I think median img makes sense since it looks pretty much like every other frame with some
# goofy stuff going on with the numbers

# Now subtract the median image from the original img_stack, going to compare that with frame 37 again
sub_stack = img_stack .- med_img
heatmap(sub_stack[:, :, 37], color=:inferno, title="frame 37 (subtracted)", size=(800, 600))
# actually using the numbers here is a good way to tell things were actually subtracted
# the median should subtract essentially '8' from the second and third digits where as 
# the first digit would be much less varied so you get some extremes left over

# vectorize the sub_stack for histo plotting
stack_vec = vec(sub_stack)
pixel_histo = histogram(stack_vec, 
                        title="remaining values after median substraction", 
                        xlabel="intensity", 
                        ylabel="frequency", 
                        xlims=[-0.10, 0.10], 
                        color=:purple, 
                        legend=false, 
                        size=(800, 600))

# As expected and hinted in the problem, wee see most of our remaining values arre sitting around zero


# Homework 6 Problem 1b
# Supposing the minimum and maximum values that matter are between -0.05 and 0.05
min_val, max_val = -0.05, 0.05

# scale the intensities
scaled_stack = clamp.(255 * (sub_stack .- min_val) / (max_val - min_val), 0, 255)
scaled_stack = round.(UInt8, scaled_stack)

heatmap(scaled_stack[:, :, 37], color=:inferno, title="frame 37 (scaled)", size=(800, 600))
# looks to highlight some details

save("homework/submitted/homework_6/images/cilia_sub_rescale_Bret.tif", scaled_stack)


# Homework 6 Problem 1c
std_img = std(img_stack, dims=3)
std_heat = heatmap(std_img[:, :, 1], color=:viridis, title="stdev of stack in time", size=(800, 600))
savefig(std_heat, "homework/submitted/homework_6/images/std_img.png")


# Homework 6 Problem 1d
kernel = Kernel.gaussian(2)

filtered_stack = similar(scaled_stack)

for i = 1:1:size(scaled_stack, 3)
    filtered_frame = imfilter(scaled_stack[:, :, i], kernel, "replicate")
    filtered_stack[:, :, i] = clamp.(round.(Int, filtered_frame), 0, 255) |> x -> convert(Array{UInt8, 2}, x)
end

save("homework/submitted/homework_6/images/test.tif", filtered_stack)