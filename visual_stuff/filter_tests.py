import cv2
import numpy as np

#my_filter should be a 3x3 array
def mean_filter(img, my_filter):
    blurred_Img = np.zeros(shape=img.shape)
    radius = int(len(my_filter)/2)
    filter_mean = radius + 1
    for i in range(len(img)):
        for j in range(len(img[i])):
            #this is applied to every pixel individual
            mean_sum = 0
            for u in range(len(my_filter)):
                for v in range(len(my_filter[u])):
                    i_hat = i - radius + u
                    j_hat = j - radius + v

                    #added condition for the border
                    if(i_hat < 0 or j_hat < 0 or i_hat >= len(img) or j_hat >= len(img[0])):
                        mean_sum += 0
                    else:
                        #print("i_hat, jhat = ", i_hat, j_hat)
                        mean_sum += img[i_hat][j_hat] * my_filter[u][v]
            #normalize
            #sum(sum(filter)) to add all the numbers by which was multiplied
            if(sum(sum(my_filter)) > 0):
                mean_sum = int(mean_sum/(sum(sum((my_filter)))))
            blurred_Img[i][j] = mean_sum
    #IMPORTANT: convert to uint not int!
    return blurred_Img.astype(np.dtype(np.uint8))

def apply_filter(img, my_filter, title):

    blurred_img = mean_filter(img, my_filter)

    cv2.imshow(title, blurred_img)


path = "E:/Programmieren/Python/learnPython/trial_and_error/visual_stuff/"
flower = cv2.imread(path + 'flower.jpg')
flower = np.array(cv2.cvtColor(flower, cv2.COLOR_RGB2GRAY))

#cv2 bild:
#bild[y][x][c]

standard_f = np.array([[1,1,1],[1,1,1],[1,1,1]])
gaussian_f = np.array([[1,2,1],[2,4,2],[1,2,1]])
invert_f = np.array([[0,0,0],[0,-1,0],[0,0,0]])
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_y = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])

cv2.imshow('Original', flower)
#apply_filter(flower, standard_f, 'Standard')
#apply_filter(flower, gaussian_f, 'Gaussian')
apply_filter(flower, sobel_x, 'x')
apply_filter(flower, sobel_y, 'y')

cv2.waitKey()
cv2.destroyAllWindows()
