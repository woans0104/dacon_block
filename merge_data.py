import pandas as pd
import numpy as np
import os
import cv2
import random
    

class merge_images:
    
    def __init__(self, df, id_col='id', image_path=None, material_max_blocks=3,
                seed=42, limit_patience = 100, board_size = 2000, x_move_ratio = 0.95,
                 y_move_ratio = 0.1, default_image_size=400, background_path=None
                ):
                
                
        self.seed = seed
        self.limit_patience = limit_patience
        self.board_size = board_size
        self.x_move_ratio = x_move_ratio
        self.y_move_ratio = y_move_ratio
        self.default_image_size = default_image_size
        self.material_max_blocks = material_max_blocks
        self.background_path = background_path
    
        self.true_labels = ['A','B','C','D','E','F','G','H','I','J']
        self.cand_dict = {1:([1]), 2:([2],[1,1]), 3:([3],[1,2],[1,1,1]), 4:([4],[1,3],[2,2],[1,1,2]),
                    5:([4,1],[3,2],[1,1,3],[1,2,2]), 6:([4,2],[3,3],[1,1,4],[1,2,3],[2,2,2]), 
                     7:([4,3],[1,2,4],[2,2,3]), 8:([4,4],[3,3,2],[4,2,2],[4,3,1]),
                    9:([4,4,1],[4,3,2]), 10:([4,4,2],[3,4,3])}
        
        
        if 'label_sum' in df.columns: 
            filter_df = df[df['label_sum']<=self.material_max_blocks]
        
        else:
            df['label_sum'] = np.sum(df.loc[:,self.true_labels], axis=1)
            filter_df = df[df['label_sum']<=self.material_max_blocks]
            
        filter_df = filter_df[filter_df['label_sum']<=self.material_max_blocks][[id_col,'label_sum']+self.true_labels]

        id_list = filter_df[id_col].tolist()
        sum_list = filter_df['label_sum'].tolist()

        label_list = []
        for _ in range(filter_df.shape[0]):
            label_list.append([])

        for t_label in self.true_labels:
            filter_list = filter_df[t_label].tolist()
            for i, value in enumerate(filter_list):
                if value==1:
                    label_list[i].append(t_label)

        for i in range(len(label_list)):
            label_list[i] = tuple(label_list[i])
        
        if image_path:
            obj = [(f'{image_path}/{id_}.jpg', sum_num, t_labels) for id_, sum_num, t_labels in zip(id_list, sum_list, label_list)]
        else:
            obj = [(f'{id_}', sum_num, t_labels) for id_, sum_num, t_labels in zip(id_list, sum_list, label_list)]
        
        
        self.pool_dict = {}
        for path, sum_num, t_labels in obj:
            if sum_num not in self.pool_dict.keys():
                self.pool_dict[sum_num] = {}

            if t_labels not in self.pool_dict[sum_num].keys():
                self.pool_dict[sum_num][t_labels] = []
            self.pool_dict[sum_num][t_labels].append(path)

        
    
    def make_new_data(self, take_num=0,
                      w_block_size=[240,250,260,270,280],
                   h_block_size=[240,250,260,270,280],
                      auto_block_size=True,
                   target_labels=None):
        
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
        #count = 0

        for right_left_ind, choice_num in enumerate(rest_cands):
            if right_left_ind>0:
                past_x_diff = past_right-past_left
                past_x_move = int(past_x_diff*self.x_move_ratio)
                past_left = past_left + past_x_move
                past_right = past_right + past_x_move
                past_median = past_left + int((past_right-past_left)/2)

            rest_num = choice_num
            except_num_set = set()
            patience = 0
            past_top, past_down, past_y_pos = 2000, 2000, 2000

            top_down_ind = 0
            patience = 0
            
            while rest_num!=0:
                if rest_num==1:
                    block_num=1
                else:
                    block_num = min(random.choice([i for i in range(1, rest_num) if i not in except_num_set]), self.material_max_blocks)
                block_labels = tuple(random.sample(rest_label_set, block_num))
                if block_labels not in self.pool_dict[block_num].keys():
                    if block_num==1:
                        continue
                    patience += 1
                    continue
                if patience==self.limit_patience:
                    patience = 0
                    except_num_set.add(block_num)
                    continue

                fig_name = random.choice(self.pool_dict[block_num][block_labels])

                new_img, mask = extract_img(fig_name)

                y_list, x_list = np.where(mask<255)
                cur_top = np.min(y_list)
                cur_down = np.max(y_list)

                cur_right = np.max(x_list)
                cur_left = np.min(x_list)
                #cur_median = int((cur_right-cur_left)/2)
                cur_median = extract_ten_percent(y_list, x_list, 'down')

                if top_down_ind==0:
                    x_move = past_left - cur_left 
                else:
                    x_move = past_median - cur_median

                y_move = past_y_pos - cur_down - 1

                new_x_list = x_list + x_move
                new_y_list = y_list + y_move

                for i in range(len(new_x_list)):
                    new_x = new_x_list[i]
                    new_y = new_y_list[i]

                    old_x = x_list[i]
                    old_y = y_list[i]
                    block_value = new_img[old_y,old_x,:]
                    all_xy_coors.append((new_y,new_x,block_value))


                # reposition
                next_left = np.min(new_x_list)
                next_right = np.max(new_x_list)
                next_top = np.min(new_y_list)
                next_down = np.max(new_y_list)


                if next_right>past_right:
                    past_left = next_left
                    past_right = next_right

                past_median = extract_ten_percent(new_y_list, new_x_list, 'up')

                past_y_pos = next_top+int((next_down-next_top)*self.y_move_ratio)

                rest_num -= block_num

                for t_label in block_labels:
                    rest_label_set.remove(t_label)

                top_down_ind += 1
        
        new_board = np.full((self.board_size,self.board_size,3),255)
        for i,(y,x,z) in enumerate(all_xy_coors):
            new_board[y,x,:] = z
        
        # resize
        x_coors, y_coors, _ = np.where(new_board!=255)
        left_x = np.min(x_coors)
        right_x = np.max(x_coors)
        down_y = np.min(y_coors)
        top_y = np.max(y_coors)
        bbox = new_board[left_x:right_x, down_y:top_y, :].astype('float32')

        w_auto_block_sw = 0
        h_auto_block_sw = 0
        
        selected_w_size = 0
        selected_h_size = 0
        
        w_bbox = bbox.shape[0]
        h_bbox = bbox.shape[1]
        if auto_block_size:
            if w_bbox>=self.default_image_size*0.7:
                w_auto_block_sw = 1
            if h_bbox>=self.default_image_size*0.7:
                h_auto_block_sw = 1
        
        w_size = w_bbox
        h_size = h_bbox
        
        if not auto_block_size or w_auto_block_sw==1:
            if isinstance(w_block_size, list):
                selected_w_size = np.random.choice(w_block_size)
            else:
                selected_w_size = w_block_size


        if not auto_block_size or h_auto_block_sw==1:
            if isinstance(h_block_size, list):
                selected_h_size = np.random.choice(h_block_size)
            else:
                selected_h_size = h_block_size

            
        w_diff_ratio = 1
        h_diff_ratio = 1
        if selected_w_size:
            w_diff_ratio = w_diff_ratio-(((selected_w_size-w_size)/w_size)*-1)
        if selected_h_size:
            h_diff_ratio = h_diff_ratio-(((selected_h_size-h_size)/h_size)*-1)
        
        if selected_w_size or selected_h_size:
            stand_ratio = min([w_diff_ratio, h_diff_ratio])
            w_size = int(w_size*stand_ratio)
            h_size = int(h_size*stand_ratio)

        bbox = cv2.resize(bbox, (h_size, w_size))
        
        h_default_size = self.default_image_size
        w_default_size = self.default_image_size
        
        
        # make background
        
        w_margin = int((w_default_size-w_size)/2)
        h_margin = int((h_default_size-h_size)/2)

        # check num
        # h_size
        h_diff = 10000
        while h_diff!=h_size:
            h_diff = h_default_size-2*h_margin
            if h_diff>h_size:
                h_default_size-=1
            elif h_diff<h_size:
                h_default_size+=1

        w_diff = 10000
        while w_diff!=w_size:
            w_diff = w_default_size-2*w_margin
            if w_diff>w_size:
                w_default_size-=1
            elif w_diff<w_size:
                w_default_size+=1
        if not self.background_path:
            back_ground = np.full((h_default_size, w_default_size,3), 255)
            back_ground[w_margin:w_default_size-w_margin, h_margin:h_default_size-h_margin,:] = bbox
        else:
            background_target = random.choice(self.background_path)
            back_ground = cv2.imread(background_target)
            back_ground = cv2.resize(back_ground, (400,400))
            #bbox_y_list, bbox_x_list, _ = np.where(bbox<255)
            new_bbox, bbox_mask = extract_img(image_name=None, img_example=bbox.astype('uint8'), edge=0)
            
            """
            img_bg = np.full((bbox.shape[0], bbox.shape[1], 3), 0).astype('uint8')
            bbox_example = bbox.astype('uint8').copy()
            bbox_example = cv2.cvtColor(bbox_example, cv2.COLOR_BGR2RGB)
            img_bg_gray=cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
            img_gray=cv2.cvtColor(bbox_example, cv2.COLOR_BGR2GRAY)

            diff_gray=cv2.absdiff(img_bg_gray,img_gray)
            diff_gray_blur = cv2.GaussianBlur(diff_gray,(5,5),0)
            diff_gray_blur = cv2.medianBlur(diff_gray_blur, 5)
            ret, bbox_mask = cv2.threshold(diff_gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            """
            bbox_y_list, bbox_x_list = np.where(bbox_mask<255)
            
            for y, x in zip(bbox_y_list, bbox_x_list):
                back_ground[w_margin+y, h_margin+x] = new_bbox[y,x,:]

            #back_ground = np.full((h_default_size, w_default_size,3), 255)
            #back_ground[w_margin:w_default_size-w_margin, h_margin:h_default_size-h_margin,:] = bbox

        return back_ground
    
    

def extract_img(image_name, edge=10, default_image_size=400, img_example=None):

    #image_name = 'TRAIN_00000'
    
    if img_example is None:
        img_example = cv2.imread(image_name)
    
    img_bg = np.full((img_example.shape[0], img_example.shape[1], 3), 0).astype('uint8')

    # 
    img_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB)
    img_bg_gray=cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    img_gray=cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)
    diff_gray=cv2.absdiff(img_bg_gray,img_gray)
    diff_gray_blur = cv2.GaussianBlur(diff_gray,(5,5),0)
    diff_gray_blur = cv2.medianBlur(diff_gray_blur, 5)
    
    ret, mask = cv2.threshold(diff_gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # edge 처리 : 블럭을 딸 때 edge 부근 일부도 블럭으로 인식되어 이를 제거함
    if edge>0:
        mask[:,:edge] = 255
        mask[:,-edge:] = 255
        mask[:edge,:] = 255
        mask[-edge:,:] = 255

    new_img = np.full((img_example.shape[0], img_example.shape[1], 3), 255).astype('uint8')

    new_img[np.where(mask==0)] = img_example[np.where(mask==0)]

    return new_img, mask
    
    
def extract_ten_percent(y_list, x_list, size=400, direct='down', percent=.1):
    
    down = np.max(y_list)
    up = np.min(y_list)
    left = np.min(x_list)
    right = np.max(x_list)
    
    y_diff = down-up
    y_ten_percent = int(y_diff*percent)

    if direct=='down':
        down = up + y_ten_percent
    else:
        up = down - y_ten_percent
    
    x_result = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        if y<=down and y>=up:
            x_result.append(x)

    x_median = np.min(x_result) + int((np.max(x_result)-np.min(x_result))/2)

    return x_median
    
    
    
def stop_code(count, limit=3):
    if count==limit:
        raise ValueError('stop by user.')
