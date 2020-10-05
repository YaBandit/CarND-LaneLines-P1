import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

show_stages = False


def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def make_points(image, avg):
    slope, yint = avg
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))

    r1 = y1 - yint
    r2 = y2 - yint

    x1 = int(r1 // slope)
    x2 = int(r2 // slope)
    return np.array([x1, y1, x2, y2])


def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)

    left_line = None
    if not np.isnan(np.min(left_avg)):
        left_line = make_points(image, left_avg)

    right_line = None
    if not np.isnan(np.min(right_avg)):
        right_line = make_points(image, right_avg)

    #left_line = make_points(image, left_avg)
    #right_line = make_points(image, right_avg)

    #right_line = make_points(image, right_avg) if not np.any.isnan(right_avg) else None

    return np.array([left_line, right_line])


def display_lines(img, lines):
    lines_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    avg_lines = average(img, lines)

    line_image = display_lines(img, avg_lines)

    res = weighted_img(line_image, img)

    return res


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #plot(lines)

    return draw_lines(line_img, lines)


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def plot(image):
    if show_stages:
        plt.imshow(image, cmap='gray')  # Remove CMAP to see it in Green
        plt.show()


def process_img(image, vertices):

    grey = grayscale(image)
    plot(grey)

    blurred = gaussian_blur(grey, 5)
    plot(blurred)

    canny_img = canny(blurred, 50, 150)
    plot(canny_img)

    # imshape = image.shape
    # vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)

    region = region_of_interest(canny_img, vertices)
    plot(region)

    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 5  # minimum number of votes (intersections in Hough grid cell)=
    # min_line_length = 40  # minimum number of pixels making up a line
    # max_line_gap = 5  # maximum gap in pixels between connectable line segments

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)=
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    lines = hough_lines(region, rho, theta, threshold, min_line_length, max_line_gap)
    plot(lines)

    res = weighted_img(lines, image) # Original image with lines added
    plot(res)

    return res


def process_and_show_img(image, vertices):

    res = process_img(image, vertices)

    plt.imshow(res, cmap='gray') # Remove CMAP to see it in Green
    plt.show()


# PROJECT CODE:

# PART 1 - Find & draw the lanes on the test images

def run_still_images():
    image1 = mpimg.imread('..\\test_images\\solidWhiteCurve.jpg')
    image2 = mpimg.imread('..\\test_images\\solidWhiteRight.jpg')
    image3 = mpimg.imread('..\\test_images\\solidYellowCurve.jpg')
    image4 = mpimg.imread('..\\test_images\\solidYellowCurve2.jpg')
    image5 = mpimg.imread('..\\test_images\\solidYellowLeft.jpg')
    image6 = mpimg.imread('..\\test_images\\whiteCarLaneSwitch.jpg')

    imshape = image1.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)

    process_and_show_img(image1, vertices)
    process_and_show_img(image2, vertices)
    process_and_show_img(image3, vertices)
    process_and_show_img(image4, vertices)
    process_and_show_img(image5, vertices)
    process_and_show_img(image6, vertices)



#run_still_images()




# PART 2 - Apply your pipeline to Example Files

# image7 = mpimg.imread('..\\solution\\image6.jpg')
# imshape_vid = image7.shape
# y_max = 459
# y_min = 634
# vertices_vid = np.array([[(200, y_min), (550, y_max), (720, y_max), (1200, y_min)]], dtype=np.int32)
#
# process_and_show_img(image7, vertices_vid)

video = cv2.VideoCapture('..\\test_videos\\challenge.mp4')


def getFrame(sec, vertices):
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    has_frames, image = video.read()

    if has_frames:

        # ORIGINALS
        #cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
        #vid_images.append(image)

        ## CONVERTED
        line_img = process_img(image, vertices)
        #cv2.imwrite("image" + str(count) + ".jpg", line_img)  # save frame as JPG file
        vid_images.append(line_img)

    return has_frames


sec = 0
frameRate = 0.05  # it will capture image in each 0.5 second
count = 1
vid_images = []

y_max = 459
y_min = 634
vertices_vid = np.array([[(200, y_min), (550, y_max), (720, y_max), (1200, y_min)]], dtype=np.int32)

success = getFrame(sec, vertices_vid)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, vertices_vid)

# Output Video

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20
out = cv2.VideoWriter('output.avi', fourcc, fps, (1280, 720))

for i in range(len(vid_images)):
    out.write(vid_images[i])
out.release()

print("Finish")
