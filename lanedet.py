import cv2 as cv
import numpy as np



def convert_slope_intercept_to_line(y1, y2 , line):
    if line is None:
        return None
    
    slope, intercept = line
    x1 = int((y1- intercept)/slope)
    y1 = int(y1)
    x2 = int((y2- intercept)/slope)
    y2 = int(y2)
    return((x1, y1),(x2, y2))

def draw_weighted_lines(img, lines, color=[255, 0, 0], thickness=2, alpha = 1.0, beta = 0.95, gamma= 0):
    mask_img = np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv.line(mask_img, *line, color, thickness)            
    return weighted_img(mask_img, img, alpha, beta, gamma)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv.addWeighted(initial_img, α, img, β, γ)

def adjust_gamma(image, gamma):
    # Build a lookup table mapping the pixel values to adjusted values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    
    # Apply the lookup table to adjust the image
    adjusted_image = cv.LUT(image, table)
    
    return adjusted_image




while 1:

   
    img = cv.imread('lane.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


    gamma_img = adjust_gamma(gray,0.5)
    hls_img = cv.cvtColor(img,cv.COLOR_BGR2HLS)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv.inRange(hls_img, lower_yellow, upper_yellow)

    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])

    white_mask = cv.inRange(hls_img, lower_white, upper_white)

    mask = cv.bitwise_or(yellow_mask, white_mask)

    colored_img = cv.bitwise_and(gamma_img, gamma_img, mask=mask)

    gauss_img =  cv.GaussianBlur(colored_img,(7,7),0)
    edges = cv.Canny(gauss_img,70,140)

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=100)
    line_image = np.copy(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


    left_lane=[]
    right_lane = []
    
    left_len=[]
    right_len=[]

    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        line_len = np.sqrt((x2-x1)**2+(y2-y2)**2)

        if slope >0 and  x1 > img.shape[1] / 2 and x2 > img.shape[1] / 2:
            left_lane.append((slope,intercept))
            left_len.append(line_len)
        elif  slope <0 and  x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
            right_lane.append((slope,intercept))
            right_len.append(line_len)


    left_avg = np.dot(left_len, left_lane)/np.sum(left_len) if len(left_len) > 0 else None
    right_avg = np.dot(right_len, right_lane)/np.sum(right_len) if len(right_len) > 0 else None

    y1 = img.shape[1]
    y2 = img.shape[1]*0.33

    left_lane = convert_slope_intercept_to_line(y1, y2, left_avg)
    right_lane = convert_slope_intercept_to_line(y1, y2, right_avg)
    
    result = draw_weighted_lines(img, [left_lane, right_lane], thickness= 10)

    cv.imshow('test1',result)

    if cv.waitKey(1)==ord('x'):
        break
cap.release()
cv.destroyAllWindows
