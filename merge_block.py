
import cv2
import numpy as np
import random
from collections import OrderedDict


class merge_images:

    def __init__(self, train_df, board_size=2000, random_seed=42):
        self.train_df = train_df
        self.board_size = board_size
        #self.random_seed = random_seed
        #random.seed(self.random_seed)
        #np.random.seed(self.random_seed)
        # self.new_board = np.full((self.board_size, self.board_size, 3), 255).astype('uint8')

    def extract_img(self, image_name):

        # image_name = 'TRAIN_00000'
        img_example = cv2.imread(image_name)
        img_bg = np.full((400, 400, 3), 0).astype('uint8')

        # Background - Gray
        img_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB)
        img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(img_bg_gray, img_gray)
        diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        diff_gray_blur = cv2.medianBlur(diff_gray_blur, 5)

        ret, mask = cv2.threshold(diff_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # previewImg("Otsu Treshold",mask,True)
        new_img = np.full((400, 400, 3), 255).astype('uint8')

        new_img[np.where(mask == 0)] = img_example[np.where(mask == 0)]

        # mask는 객체 0 배경 255 으로 구성됨

        return new_img, mask

    def extract_xy(self, mask):
        ys, xs = np.where(mask == 0)
        left_x, right_x = np.min(xs), np.max(xs)
        down_y, up_y = np.min(ys), np.max(ys)
        median_x = np.round(np.median(np.unique(xs))).astype(int)
        median_y = np.round(np.median(np.unique(ys))).astype(int)

        x_coors = [left_x, right_x, median_x]
        y_coors = [down_y, up_y, median_y]

        return x_coors, y_coors

    def make_new_data(self,
                      target_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                      block_size=[200, 250],  # min max
                      height=4, width=4, filter_horizon=True):

        new_board = np.full((self.board_size, self.board_size, 3), 255).astype('uint8')
        coor_dict = {(0, 0): {'x': 500, 'y': self.board_size}}
        count = 0
        stop_count = len(target_labels)
        cur_direct = 'up'
        right_max_x = 0
        past_i = 0

        labels = []  # label 추가
        for i in range(width):
            sw = 0
            row_min_x = self.board_size
            row_max_x = 0
            row_min_y = self.board_size
            row_max_y = 0

            for j in range(height):

                t_label = target_labels[count]
                print(t_label)

                # ------------------------------------------------------------------------------------#
                # row_num 코드 수정
                row_num = len(self.train_df[(self.train_df[t_label] == 1) & (self.train_df['label_sum'] == 1)].iloc[:, 0])
                random_row_num = random.choice(np.arange(row_num))
                image_name = self.train_df[(self.train_df[t_label] == 1) & (self.train_df['label_sum'] == 1)].iloc[
                    random_row_num, 0]
                # label 추가
                labels.append(self.train_df[(self.train_df[t_label] == 1) & (self.train_df['label_sum'] == 1)].iloc[random_row_num,
                              2:-1].tolist())

                # ------------------------------------------------------------------------------------#

                new_img, mask = self.extract_img(train_image_path + f'/{image_name}.jpg')
                x_coors, y_coors = self.extract_xy(mask)

                start_x = coor_dict[(i, j)]['x']
                start_y = coor_dict[(i, j)]['y']

                ok_y_inds, ok_x_inds = np.where(mask == 0)
                t_x_list = []
                t_y_list = []

                min_x = x_coors[0]
                min_y = y_coors[0]
                max_y = y_coors[1]

                median_x_value = int(np.round(np.median(np.unique(ok_x_inds))))

                for k in range(len(ok_x_inds)):
                    t_x = ok_x_inds[k]
                    t_y = ok_y_inds[k]

                    new_x = start_x + t_x - min_x
                    if cur_direct == 'up':
                        new_x = new_x - (median_x_value - min_x)
                    new_y = t_y + (start_y - max_y) - 1

                    t_x_list.append(new_x)
                    t_y_list.append(new_y)
                t_x_list = np.array(t_x_list)
                t_y_list = np.array(t_y_list)

                right_max_x = max([right_max_x, np.max(t_x_list)])
                new_board[(t_y_list, t_x_list)] = new_img[np.where(mask == 0)]

                # next method
                if j == 2:
                    p = np.random.uniform()
                    if p <= .5:
                        cur_direct = 'right'
                        sw = 1
                elif j == 3:
                    cur_direct = 'right'
                else:
                    cur_direct = 'up'

                if i != 0 and j != 0:
                    if cur_direct == 'up':
                        start_y += random_y_value
                    elif cur_direct == 'right':
                        start_x += random_x_value + right_max_x

                random_y_value = random.choice([50])
                random_x_value = random.choice([0])

                if cur_direct == 'up':
                    next_x = int(np.round(np.median(np.unique(t_x_list))))
                    next_y = np.min(t_y_list) + random_y_value
                elif cur_direct == 'right':
                    next_x = np.max(t_x_list) + random_x_value
                    # next_x = right_max_x+random_x_value
                    next_y = self.board_size

                row_min_x = np.min([row_min_x, np.min(t_x_list)])
                row_max_x = np.max([row_min_x, np.max(t_x_list)])
                row_min_y = np.min([row_min_y, np.min(t_y_list)])
                row_max_y = np.max([row_max_y, np.max(t_y_list)])

                if j < 3 and cur_direct == 'up':
                    coor_dict[(i, j + 1)] = {'x': next_x, 'y': next_y}
                else:
                    coor_dict[(i + 1, 0)] = {'x': next_x, 'y': next_y}

                if sw == 1:
                    break
                count += 1 # 수정
                if count>=stop_count: # 수정
                    break



            target_new_board = new_board[:, row_min_x:row_max_x, :]
            temp_fill_list = []
            for check_num in range(target_new_board.shape[0]):
                unique_num = np.unique(new_board[check_num, row_min_x:row_max_x])
                if len(unique_num) == 1:
                    if unique_num[0] == 255:
                        temp_fill_list.append(check_num)

            if temp_fill_list:
                temp_fill_list = np.array(temp_fill_list)

                # target_new_board = target_new_board[~temp_fill_list,:]
                target_new_board = np.delete(target_new_board, temp_fill_list, axis=0)
                padding = np.full((len(temp_fill_list), target_new_board.shape[1], 3), 255).astype('uint8')
                new_board[:, row_min_x:row_max_x, :] = np.r_[padding, target_new_board]

            if count == stop_count:
                break

        # resize
        x_coors, y_coors, _ = np.where(new_board != 255)
        left_x = np.min(x_coors)
        right_x = np.max(x_coors)
        down_y = np.min(y_coors)
        up_y = np.max(y_coors)
        bbox = new_board[left_x:right_x, down_y:up_y, :]

        if isinstance(block_size, list):
            size = np.random.choice(np.arange(block_size[0], block_size[1], 10))
        else:
            size = block_size
        print(size)
        bbox = cv2.resize(bbox, (size, size))

        back_ground = np.full((400, 400, 3), 255)
        margin = int((400 - size) / 2)
        back_ground[margin:400 - margin, margin:400 - margin, :] = bbox

        # lable 추가 -----------------------

        labels = np.sum(labels, axis=0)
        labels = np.clip(labels, 0, 1)
        # ----------------------------------

        return back_ground, labels





if __name__ == '__main__':
    import os
    import pandas as pd
    from utils import random_label
    default_path = os.getcwd()
    data_path = os.path.join(default_path, 'data')
    train_image_path = os.path.join(data_path, 'train')
    test_image_path = os.path.join(data_path, 'test')

    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    sums = np.sum(train_df[labels], axis=1)
    train_df['label_sum'] = sums


    num_data = 5
    new_data_dir = "train_1"
    new_train_pd = pd.DataFrame(columns=['id', 'img_path', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

    for idx in range(num_data):

        target_labels = random_label(labels, sort=True)
        print(target_labels)
        generate_obj = merge_images(train_df)
        new_image, label_final = generate_obj.make_new_data(target_labels=target_labels, block_size=[210, 250])


        new_img_id = "{}_{}".format(new_data_dir, idx)
        new_img_path = "./{}/{}.jpg".format(new_data_dir, new_img_id)
        new_data = [new_img_id,new_img_path]
        new_data.extend(label_final)

        new_train_pd.loc[new_train_pd.shape[0]] = new_data

        new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(data_path, new_img_path),new_image)

    new_train_pd.to_csv(os.path.join(data_path,"{}.csv".format(new_data_dir)), index = None)




    # save label csv











