import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


class Viewer:

    def __init__(self, image_files_root_folder, width, height, frame_sequence):
        self.image_files_root_folder = image_files_root_folder
        self.blank_image_filename = '../resources/blank.jpg'.replace('\\', '/')
        self.single_image_width = width
        self.single_image_height = height
        self.frame_sequence = frame_sequence

    def view(self, raw_clusters):

        child_clusters = {}
        for key, cluster in raw_clusters.items():
            child_clusters[key] = Viewer.get_index_array(cluster)

        child_frame_id = Viewer.select_frame_id(self.frame_sequence)

        child_cluster_image_files = {}
        for key, child_cluster in child_clusters.items():
            child_cluster_image_files[key] = self._get_image_file_names(child_cluster, child_frame_id)

        counter = 1
        for key, image_file in child_cluster_image_files.items():
            if len(image_file) > 0:
                img_cluster = self._get_image_cluster(image_file)
                Viewer.display_image_clusters(img_cluster, key, counter)
            counter += 1

        plt.show()

    def save(self, raw_clusters, root_folder):

        child_clusters = {}
        for key, cluster in raw_clusters.items():
            child_clusters[key] = Viewer.get_index_array(cluster)

        child_frame_id = Viewer.select_frame_id(self.frame_sequence)

        child_cluster_image_files = {}
        for key, child_cluster in child_clusters.items():
            child_cluster_image_files[key] = self._get_image_file_names(child_cluster, child_frame_id)

        counter = 1
        for key, image_file in child_cluster_image_files.items():
            if len(image_file) > 0:
                img_cluster = self._get_image_cluster(image_file)
                Viewer.save_image_clusters(img_cluster, key, root_folder)
            counter += 1

        plt.show()

    def _get_image_file_names(self, cluster, frame_id):
        cluster_image_files = []
        for image_id in cluster:
            folder_name = Viewer.get_folder_name(image_id)
            cluster_image_files.append(self.image_files_root_folder + folder_name + frame_id)
        return cluster_image_files

    # Compose Image files
    def _get_image_set(self, image_list):

        overflow = len(image_list) % 6

        if overflow != 0:
            for _ in range(0, (6 - overflow)):
                image_list.append(self.blank_image_filename)

        imgs = [Image.open(i) for i in image_list]

        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        # min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        min_shape = (self.single_image_width, self.single_image_height)
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        return imgs_comb

    def _get_image_cluster(self, list_im):

        if len(list_im) < 7:
            imgs_comb = self._get_image_set(list_im)

        elif len(list_im) < 13:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)

            imgs_comb = np.vstack((imgs_comb_1, imgs_comb_2))

        elif len(list_im) < 19:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:12]
            list_im_3 = list_im[12:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), imgs_comb_3))

        elif len(list_im) < 25:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:12]
            list_im_3 = list_im[12:18]
            list_im_4 = list_im[18:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((imgs_comb_3, imgs_comb_4))))

        elif len(list_im) < 31:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:12]
            list_im_3 = list_im[12:18]
            list_im_4 = list_im[18:24]
            list_im_5 = list_im[24:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((imgs_comb_3, np.vstack((imgs_comb_4, imgs_comb_5))))))

        elif len(list_im) < 37:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:12]
            list_im_3 = list_im[12:18]
            list_im_4 = list_im[18:24]
            list_im_5 = list_im[24:30]
            list_im_6 = list_im[30:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)
            imgs_comb_6 = self._get_image_set(list_im_6)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((np.vstack((imgs_comb_3, imgs_comb_4)), np.vstack((imgs_comb_5, imgs_comb_6))))))

        else:

            list_im_1 = list_im[0:6]
            list_im_2 = list_im[6:12]
            list_im_3 = list_im[12:18]
            list_im_4 = list_im[18:24]
            list_im_5 = list_im[24:30]
            list_im_6 = list_im[30:36]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)
            imgs_comb_6 = self._get_image_set(list_im_6)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((np.vstack((imgs_comb_3, imgs_comb_4)), np.vstack((imgs_comb_5, imgs_comb_6))))))

        return Image.fromarray(imgs_comb)

    @staticmethod
    def get_index_array(my_list):
        if len(my_list) == 0:
            return []
        return [x for x in map(int, my_list.strip().split(' '))]

    # Compose the frame number to be displayed
    @staticmethod
    def select_frame_id(sequence_id):
        frame_number = sequence_id * 5
        if frame_number < 10:
            frame_number = 'frame000' + str(frame_number) + '.jpg'
        elif frame_number < 100:
            frame_number = 'frame00' + str(frame_number) + '.jpg'
        else:
            frame_number = 'frame0' + str(frame_number) + '.jpg'
        return frame_number

    # Compose image files for parent and chile clusters
    @staticmethod
    def get_folder_name(video_id):
        if video_id < 10:
            return 'seq0' + str(video_id) + '/'
        else:
            return 'seq' + str(video_id) + '/'

    @staticmethod
    def save_image_clusters(img_clstr, filename, root_folder):
        img_clstr.save(root_folder + '/' + filename + '.jpg')

        # # for a vertical stacking it is simple: use vstack
        # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        # imgs_comb = Image.fromarray(imgs_comb)
        # imgs_comb.save(filename+'.jpg')

    @staticmethod
    def display_image_clusters(img_clstr, filename, plt_id):
        plt.figure(plt_id)
        plt.title(filename)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_clstr)


class SceneViewer:

    def __init__(self, image_files_root_folder, width, height):
        self.image_files_root_folder = image_files_root_folder
        self.blank_image_filename = 'E:/Projects/unsupervised/self_organizing/som_py/ikasl/ikasl_v2/resources/blank.jpg'.replace('\\', '/')
        self.single_image_width = width
        self.single_image_height = height
        self.display_image_length = 12

    def view(self, raw_clusters):

        child_clusters = {}
        for key, cluster in raw_clusters.items():
            child_clusters[key] = SceneViewer.get_index_array(cluster)

        child_cluster_image_files = {}
        for key, child_cluster in child_clusters.items():
            child_cluster_image_files[key] = self._get_image_file_names(child_cluster)

        counter = 1
        for key, image_file in child_cluster_image_files.items():
            if len(image_file) > 0:
                img_cluster = self._get_image_cluster(image_file)
                SceneViewer.display_image_clusters(img_cluster, key, counter)
            counter += 1

        plt.show()

    def _get_image_file_names(self, cluster):
        cluster_image_files = []
        for image_id in cluster:
            filename = SceneViewer.select_frame_id(image_id)
            cluster_image_files.append(self.image_files_root_folder + filename)
        return cluster_image_files

    # Compose Image files
    def _get_image_set(self, image_list):

        overflow = len(image_list) % self.display_image_length

        if overflow != 0:
            for _ in range(0, (self.display_image_length - overflow)):
                image_list.append(self.blank_image_filename)

        imgs = [Image.open(i) for i in image_list]

        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        # min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        min_shape = (self.single_image_width, self.single_image_height)
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        return imgs_comb

    def _get_image_cluster(self, list_im):

        root_size = self.display_image_length

        if len(list_im) <= root_size:
            imgs_comb = self._get_image_set(list_im)

        elif len(list_im) <= root_size*2:

            list_im_1 = list_im[0:root_size]
            list_im_2 = list_im[root_size:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)

            imgs_comb = np.vstack((imgs_comb_1, imgs_comb_2))

        elif len(list_im) <= root_size*3:

            list_im_1 = list_im[0:root_size]
            list_im_2 = list_im[root_size:root_size*2]
            list_im_3 = list_im[root_size*2:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), imgs_comb_3))

        elif len(list_im) <= root_size*4:

            list_im_1 = list_im[0:root_size]
            list_im_2 = list_im[root_size:root_size*2]
            list_im_3 = list_im[root_size*2:root_size*3]
            list_im_4 = list_im[root_size*3:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((imgs_comb_3, imgs_comb_4))))

        elif len(list_im) <= root_size*5:

            list_im_1 = list_im[0:root_size]
            list_im_2 = list_im[root_size:root_size*2]
            list_im_3 = list_im[root_size*2:root_size*3]
            list_im_4 = list_im[root_size*3:root_size*4]
            list_im_5 = list_im[root_size*4:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((imgs_comb_3, np.vstack((imgs_comb_4, imgs_comb_5))))))

        elif len(list_im) <= root_size*6:

            list_im_1 = list_im[0:root_size]
            list_im_2 = list_im[root_size:root_size*2]
            list_im_3 = list_im[root_size*2:root_size*3]
            list_im_4 = list_im[root_size*3:root_size*4]
            list_im_5 = list_im[root_size*4:root_size*5]
            list_im_6 = list_im[root_size*5:]

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)
            imgs_comb_6 = self._get_image_set(list_im_6)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((np.vstack((imgs_comb_3, imgs_comb_4)), np.vstack((imgs_comb_5, imgs_comb_6))))))

        else:

            total_imgs = len(list_im)
            width_imgs = 6

            list_im_1 = []
            list_im_2 = []
            list_im_3 = []
            list_im_4 = []
            list_im_5 = []
            list_im_6 = []

            for i in random.sample(range(0, int(total_imgs/width_imgs)), width_imgs):
                list_im_1.append(list_im[i])
            for i in random.sample(range(int(total_imgs/width_imgs), int(total_imgs/width_imgs)*2), width_imgs):
                list_im_2.append(list_im[i])
            for i in random.sample(range(int(total_imgs/width_imgs)*2, int(total_imgs/width_imgs)*3), width_imgs):
                list_im_3.append(list_im[i])
            for i in random.sample(range(int(total_imgs/width_imgs)*3, int(total_imgs/width_imgs)*4), width_imgs):
                list_im_4.append(list_im[i])
            for i in random.sample(range(int(total_imgs/width_imgs)*4, int(total_imgs/width_imgs)*5), width_imgs):
                list_im_5.append(list_im[i])
            for i in random.sample(range(int(total_imgs/width_imgs)*5, int(total_imgs/width_imgs)*6), width_imgs):
                list_im_6.append(list_im[i])

            imgs_comb_1 = self._get_image_set(list_im_1)
            imgs_comb_2 = self._get_image_set(list_im_2)
            imgs_comb_3 = self._get_image_set(list_im_3)
            imgs_comb_4 = self._get_image_set(list_im_4)
            imgs_comb_5 = self._get_image_set(list_im_5)
            imgs_comb_6 = self._get_image_set(list_im_6)

            imgs_comb = np.vstack((np.vstack((imgs_comb_1, imgs_comb_2)), np.vstack((np.vstack((imgs_comb_3, imgs_comb_4)), np.vstack((imgs_comb_5, imgs_comb_6))))))

        return Image.fromarray(imgs_comb)

    @staticmethod
    def get_index_array(my_list):
        if len(my_list) == 0:
            return []
        return [x for x in map(int, my_list.strip().split(' '))]

    # Compose the frame number to be displayed
    @staticmethod
    def select_frame_id(sequence_id):
        if sequence_id == 0:
            return 'frame00000.jpg'
        else:
            sequence_id *= 5
            if sequence_id < 10:
                frame_number = 'frame0000' + str(sequence_id) + '.jpg'
            elif sequence_id < 100:
                frame_number = 'frame000' + str(sequence_id) + '.jpg'
            elif sequence_id < 1000:
                frame_number = 'frame00' + str(sequence_id) + '.jpg'
            elif sequence_id < 10000:
                frame_number = 'frame0' + str(sequence_id) + '.jpg'
            else:
                frame_number = 'frame' + str(sequence_id) + '.jpg'
            return frame_number

    @staticmethod
    def save_image_clusters(img_clstr, filename, root_folder):
        img_clstr.save(root_folder + '/' + filename + '.jpg')

        # # for a vertical stacking it is simple: use vstack
        # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        # imgs_comb = Image.fromarray(imgs_comb)
        # imgs_comb.save(filename+'.jpg')

    @staticmethod
    def display_image_clusters(img_clstr, filename, plt_id):
        plt.figure(plt_id)
        plt.title(filename)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_clstr)

if __name__ == "__main__":

    image_files_root_folder = 'E:\Projects\scene_analyzer\\raw_data\\frames/'.replace('\\', '/')
    single_image_width = 480
    single_image_height = 360

    parent_sequence = 1

    viewer = SceneViewer(image_files_root_folder, single_image_width, single_image_height)

    child_clusters_raw = {
        "Pathway-1": "13 23 24 25 26 27 30 33 35 36 38 39 42 49 52 53 54 55 56 57 58 59 60 61 62 63 64 66 67 68 69 72 73 74 76 79 80 87 90 91 110 112 113 130 132 133 134 135 138 139 140 141 142 161",
        "Pathway-2": "17 18 19 20 22 41 46 47 65 70 71 75 128 146 147 148 149 150 151 152 153 154 155 156 157 158 159 162 165 167 169 172 174",
        "Pathway-3": "77 78 81 82 83 85 86 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 111 114 115 166 176",
        "Pathway-4": "21 28 29 31 32 34 37 40 43 44 45 48 50 51 84 88 89 125 126 127 129 131 136 137 143 144 145 160 163 164 168",
        "Pathway-5": "1 2 3 4 5 6 7 8 9 10 11 12 14 15 16 116 117 118 119 171 178 184 192",
        "Pathway-6": "120 121 122 123 124 170 173 175 177 179 180 181 182 183 185 186 187 188 189 190 191",
        "Pathway-7": "",
        "Pathway-8": "",
        "Pathway-9": "",
        "Pathway-10": ""
    }

    viewer.view(child_clusters_raw)
