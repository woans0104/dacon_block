import os
import pandas as pd
from utils import random_label
import cv2
import numpy as np
import random
from collections import OrderedDict
import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import re



class merge_images:

    def __init__(self, df, id_col='id', image_path=None, material_max_blocks=4,
                 seed=42, limit_patience=100, board_size=2000, x_move_ratio=0.95,
                 y_move_ratio=0.1, default_image_size=400):

        self.seed = seed
        self.limit_patience = limit_patience
        self.board_size = board_size
        self.x_move_ratio = x_move_ratio
        self.y_move_ratio = y_move_ratio
        self.default_image_size = default_image_size

        self.true_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.cand_dict = {1: ([1]), 2: ([2], [1, 1]), 3: ([3], [1, 2], [1, 1, 1]), 4: ([4], [1, 3], [2, 2], [1, 1, 2]),
                          5: ([4, 1], [3, 2], [1, 1, 3], [1, 2, 2]),
                          6: ([4, 2], [3, 3], [1, 1, 4], [1, 2, 3], [2, 2, 2]),
                          7: ([4, 3], [1, 2, 4], [2, 2, 3]), 8: ([4, 4], [3, 3, 2], [4, 2, 2], [4, 3, 1]),
                          9: ([4, 4, 1], [4, 3, 2]), 10: ([4, 4, 2], [3, 4, 3])}

        filter_df = df[df['label_sum'] <= material_max_blocks][[id_col, 'label_sum'] + self.true_labels]
        id_list = filter_df[id_col].tolist()
        sum_list = filter_df['label_sum'].tolist()

        label_list = []
        for _ in range(filter_df.shape[0]):
            label_list.append([])

        for t_label in self.true_labels:
            filter_list = filter_df[t_label].tolist()
            for i, value in enumerate(filter_list):
                if value == 1:
                    label_list[i].append(t_label)

        for i in range(len(label_list)):
            label_list[i] = tuple(label_list[i])

        if image_path:
            obj = [(f'{image_path}/{id_}.jpg', sum_num, t_labels) for id_, sum_num, t_labels in
                   zip(id_list, sum_list, label_list)]
        else:
            obj = [(f'{id_}', sum_num, t_labels) for id_, sum_num, t_labels in zip(id_list, sum_list, label_list)]

        self.pool_dict = {}
        for path, sum_num, t_labels in obj:
            if sum_num not in self.pool_dict.keys():
                self.pool_dict[sum_num] = {}

            if t_labels not in self.pool_dict[sum_num].keys():
                self.pool_dict[sum_num][t_labels] = []
            self.pool_dict[sum_num][t_labels].append(path)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def make_new_data(self, take_num=0,
                      block_size=[240, 250, 260, 270, 280, 290, 300, 310, 320],
                      target_labels=None,
                      viz=False):  # check_bbox

        if target_labels:
            take_num = len(target_labels)
            rest_label_set = set(random.sample(target_labels, take_num))

        else:

            # set global params
            rest_label_set = set(random.sample(self.true_labels, take_num))

        # extract block stack candidates
        rest_cands = random.choice(self.cand_dict[take_num])

        all_xy_coors = []
        past_left = 500
        past_right = 0
        past_median = 0
        # count = 0

        total_seperate_block_li = []  # check_bbox

        for right_left_ind, choice_num in enumerate(rest_cands):
            if right_left_ind > 0:
                past_x_diff = past_right - past_left
                past_x_move = int(past_x_diff * self.x_move_ratio)
                past_left = past_left + past_x_move
                past_right = past_right + past_x_move
                past_median = past_left + int((past_right - past_left) / 2)

            rest_num = choice_num
            except_num_set = set()
            patience = 0
            past_top, past_down, past_y_pos = 2000, 2000, 2000

            top_down_ind = 0
            patience = 0
            while rest_num != 0:

                block_num = random.choice([i for i in range(1, rest_num + 1) if i not in except_num_set])
                block_labels = tuple(random.sample(rest_label_set, block_num))
                if block_labels not in self.pool_dict[block_num].keys():
                    if block_num == 1:
                        continue
                    patience += 1
                    continue
                if patience == self.limit_patience:
                    patience = 0
                    except_num_set.add(block_num)
                    continue

                print(block_labels)
                fig_name = random.choice(self.pool_dict[block_num][block_labels])

                new_img, mask = extract_img(fig_name)

                y_list, x_list = np.where(mask < 255)
                cur_top = np.min(y_list)
                cur_down = np.max(y_list)

                cur_right = np.max(x_list)
                cur_left = np.min(x_list)
                # cur_median = int((cur_right-cur_left)/2)
                cur_median = extract_ten_percent(y_list, x_list, 'down')

                if top_down_ind == 0:
                    x_move = past_left - cur_left
                else:
                    x_move = past_median - cur_median

                y_move = past_y_pos - cur_down - 1

                new_x_list = x_list + x_move
                new_y_list = y_list + y_move

                seperate_block_li = []  # check_bbox
                for i in range(len(new_x_list)):
                    new_x = new_x_list[i]
                    new_y = new_y_list[i]

                    old_x = x_list[i]
                    old_y = y_list[i]
                    block_value = new_img[old_y, old_x, :]
                    all_xy_coors.append((new_y, new_x, block_value))
                    seperate_block_li.append((new_y, new_x, block_value))  # check_bbox

                total_seperate_block_li.append({'{}'.format(block_labels): seperate_block_li})  # check_bbox
                # reposition
                next_left = np.min(new_x_list)
                next_right = np.max(new_x_list)
                next_top = np.min(new_y_list)
                next_down = np.max(new_y_list)

                if next_right > past_right:
                    past_left = next_left
                    past_right = next_right

                past_median = extract_ten_percent(new_y_list, new_x_list, 'up')

                past_y_pos = next_top + int((next_down - next_top) * self.y_move_ratio)

                rest_num -= block_num

                for t_label in block_labels:
                    rest_label_set.remove(t_label)

                top_down_ind += 1

        new_board = np.full((self.board_size, self.board_size, 3), 255)
        for i, (y, x, z) in enumerate(all_xy_coors):
            new_board[y, x, :] = z

        # resize
        x_coors, y_coors, _ = np.where(new_board != 255)
        left_x = np.min(x_coors)
        right_x = np.max(x_coors)
        down_y = np.min(y_coors)
        top_y = np.max(y_coors)
        bbox = new_board[left_x:right_x, down_y:top_y, :].astype('float32')

        if isinstance(block_size, list):
            size = np.random.choice(block_size)
        else:
            size = block_size
        bbox = cv2.resize(bbox, (size, size))

        back_ground = np.full((self.default_image_size, self.default_image_size, 3), 255)
        margin = int((self.default_image_size - size) / 2)
        back_ground[margin:self.default_image_size - margin, margin:self.default_image_size - margin, :] = bbox

        bbox_coord = make_bbox_sep(total_seperate_block_li, [left_x, right_x, down_y, top_y],
                                   board_size=self.board_size,
                                   block_resize=size,
                                   default_image_size=self.default_image_size,
                                   margin=margin)

        if viz:
            viz_path = './viz'
            os.makedirs(viz_path, exist_ok=True)
            back_ground_copy = back_ground.copy().astype('uint8')
            back_ground_copy = cv2.cvtColor(back_ground_copy, cv2.COLOR_RGB2BGR)
            for label, coord in bbox_coord:
                #import ipdb;ipdb.set_trace()
                cv2.rectangle(back_ground_copy, (coord[0]), (coord[-2]), (0, 255, 0), 2)
                cv2.putText(back_ground_copy, label, (coord[0]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2)
            plt.figure(figsize=(10, 20))
            plt.imshow(back_ground_copy)
            cv2.imwrite(os.path.join(viz_path, 'test_viz_{}.jpg'.format(label)),back_ground_copy)

        return back_ground, bbox_coord


def extract_img(image_name, edge=10):
    # image_name = 'TRAIN_00000'
    img_example = cv2.imread(image_name)
    img_bg = np.full((400, 400, 3), 0).astype('uint8')

    #
    img_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB)
    img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(img_bg_gray, img_gray)
    diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    diff_gray_blur = cv2.medianBlur(diff_gray_blur, 5)

    ret, mask = cv2.threshold(diff_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # edge 처리 : 블럭을 딸 때 edge 부근 일부도 블럭으로 인식되어 이를 제거함
    mask[:, :edge] = 255
    mask[:, -edge:] = 255
    mask[:edge, :] = 255
    mask[-edge:, :] = 255

    new_img = np.full((400, 400, 3), 255).astype('uint8')

    new_img[np.where(mask == 0)] = img_example[np.where(mask == 0)]

    return new_img, mask


def extract_ten_percent(y_list, x_list, size=400, direct='down', percent=.1):
    down = np.max(y_list)
    up = np.min(y_list)
    left = np.min(x_list)
    right = np.max(x_list)

    y_diff = down - up
    y_ten_percent = int(y_diff * percent)

    if direct == 'down':
        down = up + y_ten_percent
    else:
        up = down - y_ten_percent

    x_result = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        if y <= down and y >= up:
            x_result.append(x)

    x_median = np.min(x_result) + int((np.max(x_result) - np.min(x_result)) / 2)

    return x_median


def stop_code(count, limit=3):
    if count == limit:
        raise ValueError('stop by user.')


def make_bbox_sep(total_seperate_block_li, total_block_bbox_coord, board_size, block_resize, default_image_size,
                  margin):
    # 개별 block에 대해 bbox를 구하는 함수

    # total_seperate_block_li : board_size x board_size의 흰색 이미지에 개별 block 들이 그려진 board들의 리스트
    # total_block_bbox_coord : 총 그려진 block들의 윤곽선
    # block_resize : resize될 size

    # return [['G', [[50, 241], [214, 241], [214, 350], [50, 350]]] -> label, coordinate
    # -> coord는 좌표 왼쪽상단에서 시계방향으로 정렬되어 있음.

    total_bbox = []
    for sb in total_seperate_block_li:
        new_seperate_board = np.full((board_size, board_size, 3), 255)

        block_label, block_value = list(sb.items())[0]
        block_label = re.sub(r'[^\w\s]', '', block_label).lstrip().rstrip()
        block_label = block_label.replace(" ", ",")

        for i, (y, x, z) in enumerate(block_value):
            new_seperate_board[y, x, :] = z

        bbox_sep = new_seperate_board[total_block_bbox_coord[0]:total_block_bbox_coord[1],
                   total_block_bbox_coord[2]:total_block_bbox_coord[3], :].astype('float32')

        bbox_resize = cv2.resize(bbox_sep, (block_resize, block_resize))

        back_ground_sep = np.full((default_image_size, default_image_size, 3), 255)
        back_ground_sep[margin:default_image_size - margin, margin:default_image_size - margin, :] = bbox_resize

        # contour -> bbox
        back_ground_sep_gray = cv2.cvtColor(back_ground_sep.astype('uint8'), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(~back_ground_sep_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        # import pdb;pdb.set_trace()
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # 좌표 정렬 (왼쪽상단 시계방향으로)
        coord = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        total_bbox.append([block_label, coord])

    return total_bbox


if __name__ == '__main__':

    default_path = os.getcwd()
    data_path = os.path.join(default_path, 'data')
    train_image_path = os.path.join(data_path, 'train')
    test_image_path = os.path.join(data_path, 'test')

    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    sums = np.sum(train_df[labels], axis=1)
    train_df['label_sum'] = sums

    generate_obj = merge_images(train_df, image_path='./data/train')

    test_num=5
    for i in range(test_num):
        new_img, new_coordinate = generate_obj.make_new_data(take_num=4, viz=True)



# if __name__ == '__main__':
#
#     default_path = os.getcwd()
#     data_path = os.path.join(default_path, 'data')
#     train_image_path = os.path.join(data_path, 'train')
#     test_image_path = os.path.join(data_path, 'test')
#
#     train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
#     labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
#     sums = np.sum(train_df[labels], axis=1)
#     train_df['label_sum'] = sums
#
#     num_data = 5
#     new_data_dir = "train_1"
#     new_train_pd = pd.DataFrame(columns=['id', 'img_path', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
#
#     for idx in range(num_data):
#         target_labels = random_label(labels, sort=True)
#         generate_obj = merge_images(train_df)
#         new_image, label_final = generate_obj.make_new_data(target_labels=target_labels, block_size=[240, 300])
#
#         new_img_id = "{}_{}".format(new_data_dir, idx)
#         new_img_path = "./{}/{}.jpg".format(new_data_dir, new_img_id)
#         new_data = [new_img_id, new_img_path]
#         new_data.extend(label_final)
#
#         new_train_pd.loc[new_train_pd.shape[0]] = new_data
#
#         # save image
#         new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(data_path, new_img_path), new_image)
#         print(idx)
#         print('save success')
#
#     # save label csv
#     new_train_pd.to_csv(os.path.join(data_path, "{}.csv".format(new_data_dir)), index=None)
















