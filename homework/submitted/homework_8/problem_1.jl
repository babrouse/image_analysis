###########################
# Homework 8
# Bret A. Brouse Jr. - 11.20.2024


#################################################################
# Problem 1
#################################################################

# Preamble stuff
using Images, ImageIO, ImageMorphology;

# load dang image
img_gray = load("homework/submitted/homework_8/images/Lichtenstein_imageDuplicator_1963_gray.png")

# Have to write a function

# Define a disk-shaped structuring element (manually create it as a binary array)
function disk(radius::Int)
    r2 = radius^2
    return [x^2 + y^2 <= r2 for x in -radius:radius, y in -radius:radius]
end

img_dilated = dilate(img_gray, structuring_element)