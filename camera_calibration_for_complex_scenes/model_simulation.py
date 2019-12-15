import math
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from geometry import solve2eq, ransac_line, lsm_line_fit, new_ransac_line
#from gp_estimation import rectify_image, ransac_line
from vp_est import rectify_image1
class Line:
    def __init__(self, headx, heady, footx, footy, id):
        self.headX = headx
        self.headY = heady
        self.footX = footx
        self.footY = footy
        self.id = id
        self.position = min(headx,footx) + abs(headx-footx)/2
        self.height = heady - footy

class Tracks:
    def __init__(self,line,id,heads_line,feet_line):
        self.track = []
        self.id = id
        self.heads_line = heads_line
        self.feet_line = feet_line
        self.tracks_list = []
    def add_lines(self,line):
        if line.id == self.id:
            self.track.append(line)
        else:
            self.id = line.id
            if len(self.track) > 2 :

                self.track.append(self.heads_line)
                self.track.append(self.feet_line)
                self.tracks_list.append(self.track)
                self.track = []
                self.track.append(line)



class simulation :
    def __init__(self, width, height, focal,  tilt, roll):
        self.width = width
        self.height = height
        self.focal = focal
        self.tilt = tilt
        self.roll = roll

    def find_geometry(self):
        oivi = math.tan(np.pi*self.tilt/180)*self.focal #distance between principle point and horizon
        self.pp_x = self.width / 2
        self.pp_y = self.height / 2
        oivp = (self.focal*self.focal/oivi)
        self.vp_x = self.pp_x - oivp*math.sin(np.pi*self.roll/180)
        self.vp_y = self.pp_y + oivp*math.cos(np.pi*self.roll/180)
        vl_y = self.pp_y - oivi*math.cos(np.pi*self.roll/180)
        vl_x = self.pp_x + oivi*math.sin(np.pi*self.roll/180)
        self.horizon = []
        mh = math.tan(np.pi*self.roll/180)
        self.horizon.append(mh)
        self.horizon.append(vl_y + mh*vl_x)
        print('real vanishing point:', self.vp_x, self.vp_y)

        '''
        est_roll = 180 * math.atan((self.pp_x / 2 - self.vp_x) / (self.vp_y - self.pp_y / 2)) / np.pi
        oivi = (-self.horizon[0] * height / 2) + width / 2 - self.horizon[1] / math.sqrt(
            (self.horizon[0] * self.horizon[0]) + 1)

        est_focal = math.sqrt((oivi) * (self.vp_y - (height / 2)))
        est_tilt = (180 * math.atan(oivi / est_focal) / np.pi)
        print('tilt error:', self.tilt - est_tilt, 'roll error:', est_roll - self.roll, 'focal error: ', est_focal - self.focal)
        '''
    @staticmethod
    def solve2eq(x1, y1, x2, y2):
        a = np.asarray([[x1, 1], [x2, 1]])
        b = np.asarray([y1, y2])
        m, c = np.linalg.solve(a, b)
        return m, c

    @staticmethod
    def cal_distance(p1, p2):
        return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

    def geenrate_HF_lines(self,num_inliers, stdp, stdm, num_outliers):
        self.in_lines = []
        self.out_lines = []
        self.lines = []
        self.exp = []
        new_mm = []
        #noise_p = np.random.normal(0, stdp, num_outliers)
        #noise_m = np.random.normal(0, stdm, num_outliers)
        for id in range(num_inliers) :
            seed_x = random.randint(50,self.width)
            seed_y = random.randint(50,self.height)
            try:
                ml,cl = self.solve2eq(self.vp_x, self.vp_y, seed_x, seed_y)
            except:
                continue
            length = random.randint(70,85)
            head_y = seed_y - length
            head_x = (head_y - cl)/ml
            self.in_lines.append(((head_x, head_y),(seed_x, seed_y)))
            self.lines.append(((head_x, head_y), (seed_x, seed_y)))
            new_line = Line(head_x, head_y, seed_x, seed_y, id)
            self.exp.append(new_line)
            new_mm.append(int(ml))
        aa = sum(new_mm)/len(new_mm)

        print(min(new_mm), max(new_mm))
#        noise_m =random.randint(min(new_m), max(new_m))

        for nm in range(len(new_mm)*num_outliers):
            new_m = random.randint(-30, 30)
            if abs(new_m) <= 6 :
                new_m = abs(new_m) + 6
            new_seed_x = random.randint(50, self.width)
            new_seed_y = random.randint(50, self.height)
            new_c = new_seed_y - new_m * new_seed_x
            length = random.randint(70, 85)
            head_y = new_seed_y - length
            head_x = (new_seed_x - new_c) / new_m
            self.out_lines.append(((head_x, head_y), (new_seed_x, new_seed_y)))
            self.lines.append(((head_x, head_y), (new_seed_x, new_seed_y)))


    def drow(self):
        size = 0
        img = np.zeros([self.height + size, self.width+size])
        img = cv2.rectangle(img,(int(size/2),int(size/2)), (self.width + int(size/2), self.height+ int(size/2)),(124,255,0),3)
        plt.imshow(img)
        plt.plot(self.vp_x,self.vp_y, 'ro')
        plt.plot([0,self.width+size],[self.horizon[1],self.horizon[0]*(self.width+size) + self.horizon[1]])
        for line in self.in_lines :
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],'b')
        for line in self.out_lines :
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],'r')
        plt.show()


    def run_vp_estimation(self, oivi):
         model = 'retina'
         print(oivi, self.height, self.width, self.height/(2*oivi))
         img = np.zeros([self.height, self.width ])
         x_VP, y_VP, new_lines= rectify_image1(img, self.lines, model, self.height/(2*oivi))
         roll = 180 * math.atan((self.pp_x / 2 - x_VP) / (y_VP - self.pp_y / 2)) / np.pi
         #roll = 180 * math.atan((x / 2 - x_VP) / (y_VP - y / 2)) / np.pi
         print('vanishing point error:', abs(x_VP - self.vp_x), abs(y_VP - self.vp_y))
         return roll, [x_VP, y_VP]

    def generate_tracks(self):
        #img = np.zeros([self.height + 100, self.width + 100])
        #img = cv2.rectangle(img, (50, 50), (self.width + 50, self.height + 50), (124, 255, 0), 3)
        track = Tracks(self.exp[0], 0,0,0)
        track.add_lines(self.exp[0])
        for line,line_class in zip(self.in_lines,self.exp):
            #plt.imshow(img)
            walk_direction = random.randint(5,175)*np.pi/180
            m_HL = math.tan(walk_direction)
            c_HL = line[0][1] - m_HL*line[0][0]
            #plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]])
            #plt.plot([0,600],[self.horizon[1],(self.horizon[0]*600) + self.horizon[1]])

            try:
                vl = self.solve2eq(-m_HL,c_HL,-self.horizon[0], self.horizon[1])
            except:
                continue
            try:
                m_FL, c_FL = self.solve2eq(vl[0],vl[1],line[1][0],line[1][1])
            except:
                continue
            #plt.plot([min(vl[0], line[0][0]), max(vl[0], line[0][0])],
            #         [m_HL * min(vl[0], line[0][0]) + c_HL,
            #          m_HL * max(vl[0], line[0][0]) + c_HL])
            #plt.plot([min(vl[0], line[1][0]),max(vl[0], line[1][0]) ],[m_FL * min(vl[0], line[1][0]) + c_FL,
            #         m_FL * max(vl[0], line[1][0]) + c_FL])
            #plt.plot(vl[0],vl[1], 'ro')

            track_length = random.randint(40,50)
            H_vl_distance = self.cal_distance(line[0],vl)
            step = H_vl_distance/track_length
            for _ in range(track_length):

                new_hx = line[0][0] -np.sign(m_HL)* step
                if (new_hx > line[0][0] and new_hx < vl[0]) or (new_hx < line[0][0] and new_hx > vl[0]):
                    new_hy = m_HL*new_hx + c_HL
                    new_l = self.solve2eq(new_hx,new_hy,self.vp_x,self.vp_y)
                    new_fx, new_fy = self.solve2eq(-new_l[0],new_l[1],-m_FL,c_FL)
                    #plt.plot([new_hx,new_fx],[new_hy,new_fy])
                    new_line_class = Line(new_hx,new_hy,new_fx,new_fy, line_class.id)
                    step += step
                    track.add_lines(new_line_class)
                    heads_line = Line(line[0][0], line[0][1], vl[0], vl[1], line_class.id)
                    feet_line = Line(line[1][0], line[1][1], vl[0], vl[1], line_class.id)
                    track.heads_line = heads_line
                    track.feet_line = feet_line

        #plt.show()
        #print('asdf')
        return track

    def add_noise(self, tracks, p_noise):
        #img = np.zeros([self.height + 100, self.width + 100])
        #img = cv2.rectangle(img, (50, 50), (self.width + 50, self.height + 50), (124, 255, 0), 3)
        for track in tracks.tracks_list:
            heads_line = track.pop(-1)
            feet_line = track.pop(-1)
            #plt.imshow(img)
            #plt.plot([heads_line.headX, heads_line.footX],[heads_line.headY, heads_line.footY])
            #plt.plot([feet_line.headX, feet_line.footX], [feet_line.headY, feet_line.footY])

            for line in np.copy(track):
                new_headx =  int(np.random.normal(line.headX, p_noise, 1))
                new_heady = int(np.random.normal(line.headY, p_noise, 1))
                new_footx = int(np.random.normal(line.footX, p_noise, 1))
                new_footy = int(np.random.normal(line.footY, p_noise, 1))
                new_line = Line(new_headx,new_heady,new_footx,new_footy,line.id)
                #plt.plot([new_headx, new_footx], [new_heady, new_footy])
                track.append(new_line)
            #plt.show()
            track.append(heads_line)
            track.append(feet_line)

    def estimate_horizon(self,tracks):
        img = np.zeros([self.height + 100, self.width + 100])
        img = cv2.rectangle(img, (50, 50), (self.width + 50, self.height + 50), (124, 255, 0), 3)
        plt.imshow(img)
        x_inter = []
        y_inter = []
        for track in tracks.tracks_list:
            for idx,line in enumerate(track):
                max_idx = idx
                max_height = 0
                for new_idx,new_line in enumerate(track):
                    if new_idx != idx :
                        if line.position - new_line.position > 3 :
                            if abs(new_line.height - line.height) > max_height :
                                max_idx = new_idx
                new_line = track[max_idx]
                try:
                    heads_line = self.solve2eq(line.headX,line.headY,new_line.headX,new_line.headY)
                    feet_line = self.solve2eq(line.footX,line.footY,new_line.footX, new_line.footY)
                    intersection_point = self.solve2eq(-heads_line[0], heads_line[1], -feet_line[0], feet_line[1])
                    #plt.plot(intersection_point[0], intersection_point[1], 'ro')

                    x_inter.append(intersection_point[0])
                    y_inter.append(intersection_point[1])

                except:
                    continue
        xx,yy = ransac_line(x_inter, y_inter)
        plt.plot([0, 1000], [self.horizon[1], (self.horizon[0] * 1000) + self.horizon[1]])

        x = np.asarray(xx)
        y = np.asarray(yy)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        #plt.plot([0, 1000], [c, (m * 1000) + c])
        #plt.ylim(-2000, 2000)
        #plt.xlim(-2000, 2000)

        #plt.show()


        #print('error in horizon',abs(self.horizon[0] - m), abs(self.horizon[1] -c))
        return [m, c]






tilt = 16
roll = 1
focal = 5000
width = 1920
height = 1080
for noise in range(0,32,3):
    sim1 = simulation(width, height, focal, tilt,roll)
    sim1.find_geometry()
    sim1.geenrate_HF_lines(800, noise, noise,1)
    sim1.drow()

    tracks = sim1.generate_tracks()

    #sim1.add_noise(tracks, noise)
    est_horizon = sim1.estimate_horizon(tracks)
    oivi = (-est_horizon[0] * width / 2) +  height/ 2 - est_horizon[1] / math.sqrt((est_horizon[0] * est_horizon[0]) + 1)
    #oivi = (-m * x / 2) + y / 2 - c / math.sqrt((m * m) + 1)
    #oivi = oivi - height
    est_roll, est_VP = sim1.run_vp_estimation(oivi)
    est_focal = math.sqrt((oivi) * (est_VP[1] -  (height / 2)))
    est_tilt = (180 * math.atan(oivi / est_focal) / np.pi)
    print('tilt error:', tilt - est_tilt, 'roll error:', est_roll - roll, 'focal error: ', est_focal - focal )
''''
sim1 = simulation(1000,500,600,10,2)
sim1.find_geometry()
sim1.geenrate_HF_lines(300, 10,3,1)
#sim1.drow()
#sim1.run_vp_estimation()
tracks = sim1.generate_tracks()

sim1.add_noise(tracks,20)
sim1.estimate_horizon(tracks)
for noise in range(20):
    tracks = sim1.generate_tracks()
    sim1.add_noise(tracks, noise)
    sim1.estimate_horizon(tracks)
tilt =20.4
roll = 1.12
focal = 2500
width = 2720
height = 2076
'''