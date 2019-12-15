

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.mplot3d import axes3d
from geometry import d_distance, cal_distance
from KDEpy import FFTKDE
from scipy.stats import norm

def solve2eq (x1,y1,x2,y2) :
    a = np.asarray([[x1, 1], [x2, 1]])
    b = np.asarray([y1, y2])
    m, c = np.linalg.solve(a, b)
    return m ,c
def plane_rotation(wps_f, nvp,d, pp, arr_wps, rot_theta,staris_slope, flag):
    if not flag:
        return wps_f, 0,0
    else:
        centerx = ((int(np.min(wps_f[:, 0])) + int(np.max(wps_f[:, 0]))) / 2)
        centerz = (int(np.min(wps_f[:, 2])) + int(np.max(wps_f[:, 2]))) / 2
        centery = (((- nvp[0] * (centerx - pp[0])) - (nvp[2] * centerz) + d) / nvp[1])
        center = np.array([centerx - pp[0], centery + pp[1], centerz])
        theta = - math.acos(nvp[2])
        Rx = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        thetaz = -np.pi / 2 + math.acos(nvp[0])
        # thetay = - math.acos(nvp[1])
        Rz = np.array([[math.cos(thetaz), 0, math.sin(thetaz)], [0, 1, 0], [-math.sin(thetaz), 0, math.cos(thetaz)]])
        # Ry = np.array([[math.cos(thetay), 0, math.sin(thetay)], [0, 1, 0], [ -math.sin(thetay), 0, math.cos(thetay)]])
        tra = np.copy(arr_wps)
        tra = tra - center
        tra = tra.dot(Rz)
        tra = tra.dot(Rx)
        # tra = tra.dot(Ry)
        birdeye_view = np.copy(tra)
        theta = -np.pi * rot_theta / 180
        Ry = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        birdeye_view = birdeye_view + center

        tra = tra.dot(Ry)
        thetaz = -thetaz
        Rz = np.array([[math.cos(thetaz), 0, math.sin(thetaz)], [0, 1, 0], [-math.sin(thetaz), 0, math.cos(thetaz)]])
        tra = tra.dot(Rz)
        theta = math.acos(nvp[2]) - staris_slope * np.pi / 180
        Rx = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        tra = tra.dot(Rx)
        # thetay = -thetay
        # Ry = np.array([[math.cos(thetay), 0, math.sin(thetay)], [0, 1, 0], [-math.sin(thetay), 0, math.cos(thetay)]])
        # tra = tra.dot(Ry)
        tra = tra + center

        return tra, birdeye_view, center

def create_plane (nvp,pp,x_start,x_step,x_end,z_start,z_step,z_end,scale):
    wps = []

    grid_width = 0
    grid_height = 0
    for z in range(z_start, z_end, int(z_step / scale)):
        grid_height +=1
        grid_width = 0
        for x in range(x_start, x_end, int(x_step / scale)):
            grid_width +=1
            y = (((- nvp[0] * (x - pp[0])) - (nvp[2] * z) + nvp[3]) / nvp[1])

            wps.append([x - pp[0], y + pp[1], z])
    return np.asarray(wps), [grid_width,grid_height]
def shift_plane(arr_wps,nvp,scale,height):

    head_plane1 = []
    #arr_wps[:,2] = +max(arr_wps[:,2]) - min(arr_wps[:,2]) + arr_wps[:,2]
    for i in range(arr_wps.shape[0]):

        head_plane1.append([arr_wps[i, 0] - (height / scale) * nvp[0], arr_wps[i, 1] - (height / scale) * nvp[1],
                            (arr_wps[i, 2])- (height / scale) * nvp[2]])
    return np.asarray(head_plane1)
def create_stairs(arr_wps,dims, nvp, scale,step, num_steps):
    stair = []
    new_arr = []
    for i in range(arr_wps.shape[0]):
        stair.append([arr_wps[i, 0] - (step * (int(i / dims[0])) / scale) * nvp[0],
                      arr_wps[i, 1] - (step * (int(i / dims[0])) / scale) * nvp[1],
                      (arr_wps[i, 2]) - (step * (int(i / dims[0])) / scale) * nvp[2]])
        lower_step_idx = (int(i / dims[0]) - 1)
        if lower_step_idx <=0 :
            lower_step_idx = 0
        new_arr.append([arr_wps[i, 0] - (step * lower_step_idx/ scale) * nvp[0],
                        arr_wps[i, 1] - (step * lower_step_idx/ scale) * nvp[1],
                        (arr_wps[i, 2]) - (step * lower_step_idx/ scale) * nvp[2]])
    staris = np.asarray(stair)
    new_arr = np.asarray(new_arr)
    ll = np.zeros([staris.shape[0]*2,3])
    j = 0
    for i in range(int(arr_wps.shape[0]/dims[1])):
        ll[int(j * (arr_wps.shape[0] / dims[1])):int((j + 1) * (arr_wps.shape[0] / dims[1])), :] = new_arr[int(
            i * (arr_wps.shape[0] / dims[1])):int((i + 1) * (arr_wps.shape[0] / dims[1])), :]

        j +=1
        ll[int((j) * (arr_wps.shape[0] / dims[1])):int((j + 1) * (arr_wps.shape[0] / dims[1])), :] = staris[int(
            i * (arr_wps.shape[0] / dims[1])):int((i + 1) * (arr_wps.shape[0] / dims[1])), :]
        j = j + 1



    return ll, [dims[0],dims[1]*2]
def create_unit_vector(vp, d, hieght):
    vp = np.array([vp[0] - pp[0], vp[1] - pp[1], f])
    vp1 = vp / np.sqrt((vp[0] * vp[0]) + (vp[1] * vp[1]) + (vp[2] * vp[2]))
    nvp = np.array([vp1[0], vp1[1], vp1[2], d])
    #nvp2 = np.array([vp1[0], vp1[1], vp1[2], d2])

    dd1 = abs((nvp[0] * (pp[0]) + nvp[1] * (pp[1]) + d))
    e1 = (math.sqrt(nvp[0] * nvp[0] + nvp[1] * nvp[1] + nvp[2] * nvp[2]))
    length = dd1 / e1
    scale = hieght / length
    print('SCALE1', scale)
    #dd2 = abs((nvp2[0] * (pp[0]) + nvp2[1] * (pp[1]) + d2))
    #e2 = (math.sqrt(nvp2[0] * nvp2[0] + nvp2[1] * nvp2[1] + nvp2[2] * nvp2[2]))
    #length2 = dd2 / e2
    #print("asdfasdf", length1 - length2)
    #scale2 = hieght / length2
    #print('SCALE1', scale2)
    return nvp, scale
def construct_geometry(w,h,theta_roll,theta_tilt):
    pp = [w / 2, h / 2, 0, 1]
    m_h = math.tan(np.pi * theta_roll / 180)
    oivi = f * math.tan((theta_tilt - 90) * np.pi / 180)
    # oivi = (-m * x / 2) + y / 2 - c / math.sqrt((m * m) + 1)
    print(oivi)
    c_h = math.sqrt((m_h * m_h) + 1) * ((-m_h * w / 2) + h / 2 - oivi)
    y_VP = ((f ** 2) / oivi) + h / 2
    x_VP = w / 2 - ((math.tan(theta_roll * np.pi / 180)) * (y_VP - h / 2))
    # roll = 180 * math.atan((x / 2 - x_VP) / (y_VP - y / 2)) / np.pi
    vp = [x_VP, y_VP]
    return pp,[m_h,c_h],vp, oivi


def optimization (hxp1, hyp1, fxp1, fyp1, horizon, vp) :
    mls = []
    vpn = [0,0]
    for shift in range(-3000, 3000, 100):
        vpn[0] = vp[0] + shift
        vpn[1] = vp[1] + shift
        count = 0

        H1 = []
        H_all = []
        for phx, phy, pfx, pfy in zip(hxp1, hyp1, fxp1, fyp1):
            try:
                m, c = solve2eq(pfx, pfy, phx, phy)
            except:
                continue
            try:
                m1, c1 = solve2eq(pfx, pfy, vpn[0], vpn[1])
            except:
                continue
            angle = abs(180 * (abs(math.atan(m1)) - abs(math.atan(m))) / np.pi)
            if angle < 1 :
                try:
                    vl = solve2eq(-m, c, -horizon[0], horizon[1])
                except:
                    continue
                hi = 1 - ((d_distance([phx, phy], vl) * d_distance([pfx, pfy], (vpn[0], vpn[1]))) /
                          (d_distance([pfx, pfy], vl) * d_distance([phx, phy], (vpn[0], vpn[1]))))
                count += 1
                H1.append(hi)
                H_all.append(hi)
        print(len(H1))
        ml = 0

        mean_p1 = np.asarray(H1).mean()
        for hi in H1 :
            ml += (max(0.1*mean_p1 - abs(hi - mean_p1), 0)*max(0.1*mean_p1 - abs(hi - mean_p1), 0))

        ml =ml/(mean_p1*mean_p1)
        mls.append(ml)
    print('shift', mls.index(max(mls)), max(mls), mls[20])
    plt.figure()
    plt.plot(mls, range(0, len(mls)))
    plt.show()




if __name__ == '__main__':
    w = 2000  # image width
    h = 600  # image hieght
    img = np.zeros([h, w])
    f = 807  # in pixels
    theta_tilt = 98  # in degrees
    theta_roll = 0  # in degrees
    hieght = 400  # in cm
    d = 500  # plane bais in pixels (in 3D space)

    pp, horizon, vp, oivi = construct_geometry(w, h, theta_roll, theta_tilt)
    nvp, scale = create_unit_vector(vp, d, hieght)
    print(nvp)

    ground_plane_1, gp1_dims = create_plane(nvp, pp, -1500, 50, 2500, 2000, 35, 2400, scale)
    ground_plane_2, gp2_dims = create_plane(nvp, pp, -1500, 50, 2500, 500, 50, 2000, scale)
    ground_plane_1,gp1_dims = create_stairs(np.copy(ground_plane_1),gp1_dims, nvp, scale,15, 10)
    ground_plane_1 = shift_plane(ground_plane_1,nvp,scale,130)
    ground_plane_1, _, _ = plane_rotation(ground_plane_1, nvp, d, pp, ground_plane_1, 10, 0, False)
    ground_plane_2, _, _ = plane_rotation(ground_plane_2, nvp, d, pp, ground_plane_2, 10, 0, False)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D([0, 0], [0, 1000], [0, 0], 'r')
    ax.plot3D([0, 0], [0, 0], [0, 1000], 'b')
    ax.plot3D([0, 1000], [0, 0], [0, 0], 'g')
    ax.scatter(ground_plane_1[:, 0], ground_plane_1[:, 1], ground_plane_1[:, 2], 'ro')
    ax.scatter(ground_plane_2[:, 0], ground_plane_2[:, 1], ground_plane_2[:, 2], 'b+')
    plt.show()


    fxp1 = f * (ground_plane_1[:, 0]) / ground_plane_1[:, 2]

    fyp1 = f * (ground_plane_1[:, 1]) / ground_plane_1[:, 2]

    fxp2 = f * (ground_plane_2[:, 0]) / ground_plane_2[:, 2]

    fyp2 = f * (ground_plane_2[:, 1]) / ground_plane_2[:, 2]

    head_plane1 = []
    head_plane2 = []
    for i in range(ground_plane_1.shape[0]):
        pheight = float(np.random.normal(174, 6.70, 1))
        head_plane1.append(
            [ground_plane_1[i, 0] - (pheight / scale) * nvp[0], ground_plane_1[i, 1] - (pheight / scale) * nvp[1],
             ground_plane_1[i, 2] - (pheight / scale) * nvp[2]])
    for i in range(ground_plane_2.shape[0]):
        pheight = float(np.random.normal(174, 6.7, 1))
        # print(height)
        head_plane2.append(
            [ground_plane_2[i, 0] - (pheight / scale) * (nvp[0]), ground_plane_2[i, 1] - (pheight / scale) * nvp[1],
             ground_plane_2[i, 2] - (pheight / scale) * nvp[2]])
    head_plane1 = np.asarray(head_plane1)
    head_plane2 = np.asarray(head_plane2)
    hxp1 = f * (head_plane1[:, 0] / head_plane1[:, 2])
    hyp1 = f * (head_plane1[:, 1] / head_plane1[:, 2])
    hxp2 = f * (head_plane2[:, 0] / head_plane2[:, 2])
    hyp2 = f * (head_plane2[:, 1] / head_plane2[:, 2])
    hxp1 = hxp1 + pp[0]
    hxp2 = hxp2 + pp[0]
    fxp1 = fxp1 + pp[0]
    fxp2 = fxp2 + pp[0]
    hyp1 = hyp1 + pp[1]
    hyp2 = hyp2 + pp[1]
    fyp1 = fyp1 + pp[1]
    fyp2 = fyp2 + pp[1]
    fxp1_re = np.reshape(fxp1, [gp1_dims[1], gp1_dims[0]])
    fyp1_re = np.reshape(fyp1, [gp1_dims[1], gp1_dims[0]])
    fxp2_re = np.reshape(fxp2, [gp2_dims[1], gp2_dims[0]])
    fyp2_re = np.reshape(fyp2, [gp2_dims[1], gp2_dims[0]])

    plt.imshow(img)
    plt.plot(fxp1, fyp1, 'ro')
    plt.plot(vp[0], vp[1], 'ro')
    plt.plot(fxp2, fyp2, 'bo')
    plt.plot([fxp1[:], hxp1[:]], [fyp1[:], hyp1[:]], 'b', linewidth=1)

    for i in range(gp1_dims[1]):
        for j in range(gp1_dims[0]):
            plt.plot(fxp1_re[i, :], fyp1_re[i, :], 'w', linewidth=1)
            plt.plot(fxp1_re[:, j], fyp1_re[:, j], 'w', linewidth=1)
    for i in range(gp2_dims[1]):
        for j in range(gp2_dims[0]):
            plt.plot(fxp2_re[i, :], fyp2_re[i, :], 'w', linewidth=1)
            plt.plot(fxp2_re[:, j], fyp2_re[:, j], 'w', linewidth=1)

    plt.plot([fxp2[:], hxp2[:]], [fyp2[:], hyp2[:]], 'g', linewidth=1)
    xx = list(range(-50, img.shape[1] + 50))
    yy = [horizon[0] * x + horizon[1] for x in xx]
    print(horizon[0], horizon[1])
    plt.plot(xx, yy)
    plt.plot([fxp2_re[int(gp2_dims[1] / 2), int(gp2_dims[0] / 2)],
              fxp2_re[int(gp2_dims[1] / 2) + 1, int(gp2_dims[0] / 2)]],
             [fyp2_re[int(gp2_dims[1] / 2), int(gp2_dims[0] / 2)],
              fyp2_re[int(gp2_dims[1] / 2) + 1, int(gp2_dims[0] / 2) + 1]],
             'r', linewidth=3)

    # plt.plot([fxp2_re[int(dims2[1]/2),int(dims2[0]/2)],fxp2_re[int(dims2[1]/2)+1,int(dims2[0]/2)+1]],[fyp2_re[int(dims2[1]/2),int(dims2[0]/2)],fyp2_re[int(dims2[1]/2),int(dims2[0]/2)+1]],'b',linewidth=3)

    plt.show()

    count = 0

    H1 = []
    H_all = []
    for phx, phy, pfx, pfy in zip(hxp1, hyp1, fxp1, fyp1):
        try:
            m, c = solve2eq(pfx, pfy, phx, phy)
        except:
            continue
        try:
            m1, c1 = solve2eq(pfx, pfy, vp[0], vp[1])
        except:
            continue
        angle = abs(180 * (abs(math.atan(m1)) - abs(math.atan(m))) / np.pi)

        try:
            vl = solve2eq(-m, c, -horizon[0], horizon[1])
        except:
            continue
        hi = 1 - ((d_distance([phx, phy], vl) * d_distance([pfx, pfy], (vp[0], vp[1]))) /
                  (d_distance([pfx, pfy], vl) * d_distance([phx, phy], (vp[0], vp[1]))))
        count += 1
        H1.append(hi)
        H_all.append(hi)

    H2 = []
    for phx, phy, pfx, pfy in zip(hxp2, hyp2, fxp2, fyp2):
        try:
            m, c = solve2eq(pfx, pfy, phx, phy)
        except:
            continue
        try:
            m1, c1 = solve2eq(pfx, pfy, vp[0], vp[1])
        except:
            continue
        angle = abs(180 * (abs(math.atan(m1)) - abs(math.atan(m))) / np.pi)

        # print(angle)
        try:
            vl = solve2eq(-m, c, -horizon[0], horizon[1])
        except:
            continue
        # plt.plot(vl[0], vl[1], 'ro')
        # plt.plot([vl[0],vp[0]], [vl[1],vp[1]])

        hi = 1 - ((d_distance([phx, phy], vl) * d_distance([pfx, pfy], (vp[0], vp[1]))) /
                  (d_distance([pfx, pfy], vl) * d_distance([phx, phy], (vp[0], vp[1]))))
        count += 1
        H2.append(hi)
        H_all.append(hi)

    mean_p1 = np.asarray(H1).mean()
    std_p1 = np.asarray(H1).std()
    print('fisrt plane: ', std_p1, mean_p1)
    plt.hist(np.asarray(H1), bins=50)
    plt.xlim([-1, 2])
    plt.show()
    mean_p2 = np.asarray(H2).mean()
    std_p2 = np.asarray(H2).std()
    print('second plane:', std_p2, mean_p2)

    plt.hist(np.asarray(H2), bins=50)
    plt.xlim([-1, 2])
    plt.show()

    mean_overall = np.asarray(H_all).mean()
    std_overall = np.asarray(H_all).std()
    print('overall:', std_overall, mean_overall)

    fig, axs = plt.subplots(4, 1)
    axs[0].hist(np.asarray(H1), bins=50)
    axs[0].set_title('stairs histogram')
    axs[0].set_xlim(0, 2)
    axs[1].hist(np.asarray(H2), bins=10)
    axs[1].set_title('ground plane histogram')
    axs[1].set_xlim(0, 2)
    axs[2].hist(np.asarray(H_all), bins=50)
    axs[2].set_title('overall histogram')
    axs[2].set_xlim(0, 2)

    rr = [x / 2000 for x in range(0, 4000)]

    gpall = [math.exp(-(x - mean_overall) * (x - mean_overall) / (2 * std_overall ** 2)) / math.sqrt(
        2 * np.pi * std_overall ** 2) for x in rr]
    g_p1 = [math.exp(-(x - mean_p1) * (x - mean_p1) / (2 * std_p1 ** 2)) / math.sqrt(2 * np.pi * std_p1 ** 2) for x
            in
            rr]
    g_p2 = [math.exp(-(x - mean_p2) * (x - mean_p2) / (2 * std_p2 ** 2)) / math.sqrt(2 * np.pi * std_p2 ** 2) for x
            in
            rr]
    # fig, ax = plt.subplots()

    axs[3].plot(rr, gpall, 'k', label='Gaussian of all relative heights')
    axs[3].plot(rr, g_p1, 'r', label='Gaussian of relative heights on the lower plane')
    axs[3].plot(rr, g_p2, 'b', label='Gaussian of relative heights on stairs')
    axs[3].set_xlim(0, 2)

    # legend = ax.legend(loc=1, shadow=False, fontsize='small')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

    estimator = FFTKDE(kernel='gaussian', bw=0.02)  # silverman
    x, y = estimator.fit(H_all, weights=None).evaluate()

    plt.plot(x, y, label='KDE estimate')
    plt.show()

    print('camera height :', hieght, 'estimated camera height :', 174 / mean_p1, 'estimated camera height :',
          174 / mean_p2, 'estimated camera height :', 174 / mean_overall)
    optimization(list(hxp1)+list(hxp2), list(hyp1)+list(hyp2), list(fxp1)+list(fxp2), list(fyp1)+list(fyp2), horizon, vp)
    ######### Using otsu method ##########
    bins = np.linspace(0, 2, 50)
    histogram, bins = np.histogram(H_all, bins=bins)
    props = histogram / len(H_all)
    t = 0.8
    t_idx = np.where(bins < t)[-1][-1]

    plt.hist(H_all, bins=50)
    plt.show()
    plt.figure(figsize=(6, 4))
    minimum = 0
    for t_idx, t in enumerate(bins[:-2]):
        g1 = props[:t_idx + 1]
        q1 = sum(g1)
        u1 = sum(g1 * bins[:t_idx + 1]) / q1

        g2 = props[t_idx:]
        q2 = sum(g2)
        u2 = sum(g2 * bins[t_idx + 1:]) / q2
        bv = q1 * (1 - q1) * (u1 - u2) * (u1 - u2)
        if bv > minimum:
            threshold = t
            minimum = bv
            print(threshold)
    P1_H = []
    P2_H = []
    for  h in  H_all:

        if h < threshold:
            # plt.plot(pf0,pf1,'ro')

            P1_H.append(h)
        else:
            # plt.plot(pf0,pf1,'bo')

            P2_H.append(h)
    plt.hist(P1_H, bins=50)
    plt.show()
    plt.figure(figsize=(6, 4))
    minimum = 0
    plt.imshow(img)
    plt.show()
    std_p1 = np.asarray(P1_H).std()
    mean_p1 = np.asarray(P1_H).mean()
    std_p2 = np.asarray(P2_H).std()
    mean_p2 = np.asarray(P2_H).mean()
    std_p = np.asarray(H_all).std()
    mean_p = np.asarray(H_all).mean()
    print('lower plane height:', 174 / mean_p1, 'heigher plane height:', 174 / mean_p2, 'difference : ',
          174 / (mean_p2) - 174 / (mean_p1))

#########################################333



