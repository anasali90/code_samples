import numpy as np
import math
from skimage.measure import LineModelND, ransac

def solve2eq (x1,y1,x2,y2) :
    a = np.asarray([[x1, 1], [x2, 1]])
    b = np.asarray([y1, y2])
    m, c = np.linalg.solve(a, b)
    return m ,c

def cal_distance ( p1, p2) :
    return math.sqrt(math.pow((p1[0] - p2[0]),2) +math.pow((p1[1] - p2[1]),2) )
def d_distance (p1,p2):
    distance = math.sqrt(math.pow((p1[0] - p2[0]),2) +math.pow((p1[1] - p2[1]),2) )
    if p1[1]<p2[1]:
        return distance
    else:
        return -distance

def ransac_line(x_intersection, y_intersection, thersh=5):
    intersection_points = [x_intersection, y_intersection]
    intersection_points = np.asarray(intersection_points)
    model_robust, inliers = ransac(intersection_points.transpose(), LineModelND, min_samples=2,
                                   residual_threshold=thersh, max_trials=1000)
    l = []
    ll = []
    for j, i in enumerate(inliers):
        if i:

            l.append(intersection_points[0,j])
            ll.append(intersection_points[1,j])
    return l, ll
def new_ransac_line (x_intersection, y_intersection, thersh, iters_num, img) :
    height = img.shape[0]
    max_value = 0
    maxi = 0
    i = 0
    while i < iters_num:
        idx1 = np.random.randint(0,len(x_intersection))
        idx2 = np.random.randint(0,len(x_intersection))
        new_x_intersection= []
        new_y_intersection = []

        try:
            m,c = solve2eq(x_intersection[idx1],y_intersection[idx1], x_intersection[idx2], y_intersection[idx2])
        except:
            continue

        if abs(m) < 0.08 :
            distances = abs((-m * np.asarray(x_intersection) / 2 + (np.asarray(y_intersection) / 2 - c)) / math.sqrt((m * m) + 1))
            thresholding  = (distances<thersh)*1
            votes = (thresholding).sum()
            indecies = np.where(thresholding==1)
            i += 1
            if votes > maxi :

                maxi = votes
                max_x = np.asarray(x_intersection)[indecies]
                max_y = np.asarray(y_intersection)[indecies]




    max_m , max_c = lsm_line_fit(max_x, max_y)

    return max_m,max_c, max_x, max_y



def lsm_line_fit (x_intersection,y_intersection):
    x = np.asarray(x_intersection)
    y = np.asarray(y_intersection)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m,c