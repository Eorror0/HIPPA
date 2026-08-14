
import os, sys
from os.path import join
import h5py
import math
from math import floor
import pdb
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore
import time

import nmslib
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import convert

import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
import random
from PIL import Image

import dgl

from skimage.measure import regionprops
import os.path as osp
import pickle

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage import io
import numpy as np
from tqdm import trange

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, feature,i):

        self.update(h, w, feature,i)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, feature,i):
        self.h = h
        self.w = w

        self.feature=feature
        self.i = i

    def __str__(self):

        return "{},{}:{} ".format(self.h, self.w, self.feature,self.i)

    def __repr__(self):
        return self.__str__()

class SLICProcessor(object):
    @staticmethod
    def open_image(path):

        return np.load(path,allow_pickle=True)

    @staticmethod
    def save_lab_image(path, lab_arr):

        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w, i):
        h = int(h)
        w = int(w)

        return Cluster(h, w, self.features[i],i)

    def __init__(self,h5_path,slide_name,slic_path,cancer_type,size,weight,k):

        self.h5_path  = h5_path
        self.slide_name = slide_name
        self.slic_path = slic_path
        self.cancer_type = cancer_type
        slide_h5_path = os.path.join(h5_path,slide_name+'.h5')
        wsi_h5=h5py.File(slide_h5_path,"r")
        self.features = wsi_h5['features']
        self.coords = wsi_h5['coords']
        features_array = self.features[:]
        coords_array = self.coords[:]
        self.size = size
        self.weight = weight

        self.points = np.concatenate((features_array, coords_array), axis=1)

        self.coords_dict = {tuple(coord): index for index, coord in enumerate(coords_array.tolist())}

        self.patch_size = 512

        self.N =   len(wsi_h5['features'])
        self.K= int(math.sqrt(self.N)*k)
        self.M= int(math.sqrt(self.N))

        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full(self.N , np.inf)

    def auto_clusters(self):
        import cv2
        import openslide

        slide_path = '/home/data/mntdata/data0/WSI/{}/{}.svs'.format(self.cancer_type.upper(),self.slide_name)
        slide = openslide.open_slide(slide_path)
        stitch_path = '/home/data/mntdata/data0/CLAM_patch_20x/{}/stitches/{}.jpg'.format(self.cancer_type.upper(),self.slide_name)
        image = cv2.imread(stitch_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray_image > 20, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]

        svs_width, svs_height = slide.level_dimensions[0]
        cv2_height, cv2_width, _ = image.shape

        width_ratio = svs_width / cv2_width
        height_ratio = svs_height / cv2_height

        total_area = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
        total_point = self.K
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            num_points = int(total_point * ( area / total_area))

            num_points = max(num_points, 1)
            contour_pixels = []
            for i in range(x, x + w):
                for j in range(y, y + h):

                    if cv2.pointPolygonTest(cnt, (i, j), False) >= 0:
                        contour_pixels.append((i, j))

            selected_pixels = random.sample(contour_pixels, num_points)
            for pixel in selected_pixels:
                x, y = pixel
                x_scaled = int(x * width_ratio)
                y_scaled = int(y * height_ratio)
                h,w = min(self.coords, key=lambda coord: abs(coord[0] - x_scaled) + abs(coord[1] - y_scaled))
                i = self.coords_dict.get((h, w), None)
                self.clusters.append(self.make_cluster(h,w,i))

    def init_clusters(self):
        indices = np.random.choice(self.coords.shape[0], self.K,replace=False)
        for i in indices:
            h,w =self.coords[i,]
            self.clusters.append(self.make_cluster(h,w,i))

    def find_index(self, x, y):

        for idx, [x_coord, y_coord] in enumerate(self.coords_list):
            if x_coord == x and y_coord == y:
                return idx

    def get_neighbors(self, h, w):
        neighbors = []

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        coords_array = self.coords[:]
        for dh, dw in directions:
            nh, nw = h + dh * self.patch_size, w + dw * self.patch_size

            neighbors_idx = self.coords_dict.get((nh, nw), None)
            if neighbors_idx is not None:
                neighbors.append(neighbors_idx)
            else:
                neighbors.append(-1)
        return neighbors

    def get_gradient(self, centre,rd):
        gradient=0
        for i in range(1023):
            gradient+=math.fabs((self.features[rd][i] - self.features[centre][i]))
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            neighbors = self.get_neighbors(cluster.h, cluster.w)

            cluster_gradient = float('inf')
            for neighbor_idx in neighbors:
                if neighbor_idx != -1:
                    new_gradient = self.get_gradient(cluster.i, neighbor_idx)

                    if new_gradient < cluster_gradient:
                        cluster.update(self.coords[neighbor_idx][0], self.coords[neighbor_idx][1],self.features[neighbor_idx],neighbor_idx)
                        cluster_gradient = new_gradient

    def assignment_planA(self):
        for cluster in self.clusters:
            cycle = 0
            for h in range(cluster.h - 2 * self.patch_size * self.S, cluster.h + 2 * self.patch_size * self.S,self.patch_size):

                for w in range(cluster.w - 2 *self.patch_size* self.S, cluster.w + 2 * self.patch_size * self.S,self.patch_size):

                    idx =  self.coords_dict.get((h, w), None)
                    if idx is not  None:
                        feature = self.features[idx]
                    else:
                        continue
                    feature_len=len(feature)

                    pow_sum=0
                    for i in range(feature_len):
                        pow_sum+=math.pow(feature[i]-cluster.feature[i],2)
                    Dc = math.sqrt(pow_sum)
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[idx]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[idx] = D
                cycle = cycle + 1

    def assignment_planB(self):
        result = []
        for cluster in self.clusters:
            idx =  self.coords_dict.get((cluster.h, cluster.w), None)
            center_with_hw = np.concatenate([np.expand_dims(self.features[idx], axis=0), np.array([[cluster.h, cluster.w]])], axis=1)
            distances = self.points - center_with_hw
            features_norm = np.linalg.norm(distances[:, :1024], axis=1)

            hw_norm = np.linalg.norm(distances[:, 1024:], axis=1)/self.patch_size

            weights = np.mean(np.stack((features_norm, self.weight*hw_norm)), axis=0)
            result.append(weights)

        matrix =  np.vstack(result)
        min_indices = np.argmin(matrix, axis=0)

        i = 0
        for col, row in enumerate(min_indices):
            min_value = matrix[row, col]
            self.dis[i] = min_value
            self.label[tuple(self.coords[i])] = self.clusters[row]
            self.clusters[row].pixels.append(tuple(self.coords[i]))
            i = i + 1

    def assignment_planB_gpu(self):

        result = []
        for cluster in self.clusters:
            idx = self.coords_dict.get((cluster.h, cluster.w), None)
            if idx is not None:
                center_with_hw = torch.cat([self.features[idx].unsqueeze(0), torch.tensor([[cluster.h, cluster.w]]).cuda()], dim=1)
                distances = self.points - center_with_hw
                features_norm = torch.norm(distances[:, :1024], dim=1)
                hw_norm = torch.norm(distances[:, 1024:], dim=1) / self.patch_size
                weights = torch.mean(torch.stack((features_norm, self.weight*hw_norm)), dim=0)
                result.append(weights)

        matrix = torch.stack(result)
        min_indices = torch.argmin(matrix, dim=0)
        i = 0
        for col, row in enumerate(min_indices):
            min_value = matrix[row, col]
            self.dis[i] = min_value
            self.label[tuple(self.coords[i])] = self.clusters[row]
            self.clusters[row].pixels.append(tuple(self.coords[i]))

    def assignment_planC_gpu(self):
        result = []
        for cluster in self.clusters:
            idx = self.coords_dict.get((cluster.h, cluster.w), None)
            if idx is not None:

                cluster_h = cluster.h
                cluster_w = cluster.w

                center_features = self.features[idx].unsqueeze(0)
                features_diff = self.points[:, :1024] - center_features
                features_norm = torch.norm(features_diff, dim=1)

                points_hw = self.points[:, 1024:]
                points_h = points_hw[:, 0]
                points_w = points_hw[:, 1]

                r_cluster = torch.sqrt(cluster_w ** 2 + cluster_h ** 2)
                theta_cluster = torch.atan2(cluster_h, cluster_w)

                r_points = torch.sqrt(points_w ** 2 + points_h ** 2)
                theta_points = torch.atan2(points_h, points_w)

                delta_theta = theta_points - theta_cluster
                cos_delta_theta = torch.cos(delta_theta)
                d_squared = r_cluster ** 2 + r_points ** 2 - 2 * r_cluster * r_points * cos_delta_theta
                d = torch.sqrt(d_squared)
                hw_norm = d / self.patch_size

                weights = torch.mean(torch.stack((features_norm, hw_norm)), dim=0)
                result.append(weights)

        matrix = torch.stack(result)
        min_indices = torch.argmin(matrix, dim=0)
        i = 0
        for col, row in enumerate(min_indices):
            min_value = matrix[row, col]
            self.dis[i] = min_value
            self.label[tuple(self.coords[i])] = self.clusters[row]
            self.clusters[row].pixels.append(tuple(self.coords[i]))
            i += 1

    def assignment_planC(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        result = []

        for cluster in self.clusters:
            idx = self.coords_dict.get((cluster.h, cluster.w), None)

            center_feature = torch.tensor(self.features[idx], device=device).unsqueeze(0)
            center_hw = torch.tensor([[cluster.h, cluster.w]], device=device)

            center_with_hw = torch.cat([center_feature, center_hw], dim=1)

            points_tensor = torch.tensor(self.points, device=device)

            distances = points_tensor - center_with_hw

            features_norm = torch.linalg.norm(distances[:, :1024], dim=1)

            hw_norm = torch.linalg.norm(distances[:, 1024:], dim=1) / self.patch_size

            weights = torch.mean(torch.stack((features_norm, hw_norm)), dim=0)

            result.append(weights)

        matrix = torch.vstack(result).to(device)

        min_indices = torch.argmin(matrix, dim=0)

        i = 0
        for col, row in enumerate(min_indices):
            min_value = matrix[row, col].item()
            self.dis[i] = min_value
            self.label[tuple(self.coords[i])] = self.clusters[row]
            self.clusters[row].pixels.append(tuple(self.coords[i]))
            i += 1

    def find_center_coordinate(coordinates, center_x, center_y):
        center_coordinate = None
        min_distance = float('inf')

        for coord in coordinates:
            x, y = coord
            D = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            if D < min_distance:
                min_distance = D
                center_coordinate = coord

        return center_coordinate

    def update_cluster(self):

        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            if len(cluster.pixels)>0:
                for p in cluster.pixels:
                    sum_h += p[0]
                    sum_w += p[1]
                    number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                h,w = min(self.coords, key=lambda coord: abs(coord[0] - _h) + abs(coord[1] - _w))
                i = self.coords_dict.get((h, w), None)
                cluster.update(h, w, self.features[i],i)

    def secondary_clustering(self, center2pixel, n_clusters_per_patch=3):

        secondary_centers = []
        patch_membership = {}
        center_to_secondary = {}

        print(f"开始二次聚类，每个patch聚为{n_clusters_per_patch}类...")

        for center_idx, (center_coord, pixels) in enumerate(center2pixel.items()):
            if len(pixels) < n_clusters_per_patch:

                for pixel in pixels:
                    pixel_idx = self.coords_dict[pixel]
                    secondary_centers.append(self.features[pixel_idx])
                    patch_membership[pixel] = len(secondary_centers) - 1
                center_to_secondary[center_coord] = list(range(len(secondary_centers) - len(pixels), len(secondary_centers)))
                continue

            patch_features = []
            for pixel in pixels:
                pixel_idx = self.coords_dict[pixel]
                patch_features.append(self.features[pixel_idx])

            patch_features = np.array(patch_features)

            kmeans = MiniBatchKMeans(n_clusters=min(n_clusters_per_patch, len(pixels)),
                                   random_state=42)
            secondary_labels = kmeans.fit_predict(patch_features)

            secondary_start_idx = len(secondary_centers)
            center_to_secondary[center_coord] = []

            for cluster_id in range(kmeans.n_clusters):

                cluster_center = kmeans.cluster_centers_[cluster_id]
                secondary_centers.append(cluster_center)
                center_to_secondary[center_coord].append(secondary_start_idx + cluster_id)

            for pixel, label in zip(pixels, secondary_labels):
                patch_membership[pixel] = secondary_start_idx + label

        print(f"二次聚类完成，共生成{len(secondary_centers)}个特征中心")
        return secondary_centers, patch_membership, center_to_secondary

    def iterate_times(self):
        start = time.time()
        stitch_path = '/home/data/mntdata/data0/CLAM_patch_20x_{}/{}/stitches/{}.jpg'.format(self.size,self.cancer_type.upper(),self.slide_name)
        if not os.path.isfile(stitch_path):
            self.init_clusters()
        else:

            self.init_clusters()
        for i in range(len(self.clusters)):
            if list(self.clusters)[i].h<0 or list(self.clusters)[i].w<0:
                print(i)
        self.move_clusters()

        end = time.time()

        mytimes=10
        print("init finish,time:",start-end)

        for i in trange(mytimes):
            start = time.time()
            self.assignment_planB()
            end = time.time()
            print("assignment finish,time:",start-end)
            self.update_cluster()

        center2pixel={}
        all_coords_list=[tuple([self.clusters[i].h,self.clusters[i].w,self.clusters[i].i]) for i in range(len(self.clusters))]

        all_coords_set=set(all_coords_list)
        duplicates = [x for x in all_coords_set if all_coords_list.count(x) > 1]

        for i in range(len(self.clusters)):
            coord=tuple([self.clusters[i].h,self.clusters[i].w])
            center2pixel[coord]=self.clusters[i].pixels
            if len(self.clusters[i].pixels)==0:
                center2pixel[coord]=[coord]

        center2pixel_path=os.path.join(self.slic_path,self.slide_name,self.slide_name+'_center2pixel.pkl')
        with open(center2pixel_path, 'wb') as f:
            pickle.dump(center2pixel, f)
        print(self.slide_name," CenterPixel saved")

class MakeGraph(object):
    def __init__(self,h5_path,slide_name,slic_path):
        self.h5_path  = h5_path
        self.slide_name = slide_name
        self.slic_path = slic_path
        slide_h5_path = os.path.join(h5_path,slide_name+'.h5')
        wsi_h5=h5py.File(slide_h5_path,"r")
        self.features = wsi_h5['features']
        self.coords = wsi_h5['coords']
        coords_array = self.coords[:]

        self.coords_dict = {tuple(coord): index for index, coord in enumerate(coords_array.tolist())}

    def MakePtFile(self):
        mode = 1
        center2pixel_path = os.path.join(self.slic_path,self.slide_name,self.slide_name+'_center2pixel.pkl')
        with open(center2pixel_path, 'rb') as f:
            center2pixel = pickle.load(f)

        features_stack = []
        for idx, (center, pixels) in enumerate(center2pixel.items()):
            a,b = center
            cluster_feature = []
            for pixel in pixels:
                cluster_feature.append(self.features[self.coords_dict[(pixel[0],pixel[1])]])
            stacked_feature=np.stack(cluster_feature,axis=0)
            stacked_feature_avg=np.mean(stacked_feature,axis=0)
            features_stack.append(stacked_feature_avg)
        stacked_features=np.stack(features_stack,axis=0)

        center_list = list(center2pixel.keys())
        edges = []

        if mode == 0:
            for i in range(len(center_list)):
                distances = []
                for j in range(i + 1, len(center_list)):
                    a_x, a_y = center_list[i]
                    b_x, b_y = center_list[j]
                    dist = math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)

                    distances.append((dist, j))

                distances.sort()

                for dist, j in distances[:6]:
                    edges.append((i, j))

        if mode == 1:
            for i in range(len(center_list)):
                for j in range(len(center_list)):
                    if i != j:
                        edges.append((i, j))
        else:

            for i in range(len(center_list)):
                for j in range(i + 1, len(center_list)):
                    edges.append((i,j))

        edge_spatial=np.array(edges).T

        centers = center2pixel.keys()
        from torch_geometric.data import Data as geomData
        G = geomData(x = torch.tensor(stacked_features,dtype=torch.float),
                edge_index = torch.tensor(edge_spatial,dtype=torch.long),
                centroid = torch.tensor(list(centers),dtype=torch.float32))
        graph_path=os.path.join(self.slic_path,self.slide_name,self.slide_name+'.pt')
        torch.save(G, graph_path)
        print(self.slide_name," MakePtGraph saved")

if __name__ == '__main__':

    p = SLICProcessor()
    p.iterate_times()
