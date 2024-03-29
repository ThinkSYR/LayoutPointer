"""
xfund data format visual
"""
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import json
import random

from tqdm import tqdm
from color import ncolors
random.seed(2023)

label_path  = r"./data/xfund_funsd_all/all.val.json"
img_dir     = r"./data/images"
save_dir    = r"./visual/all"
os.makedirs(save_dir, exist_ok=True)

max_visual  = 300
with open(label_path, 'r', encoding='utf-8') as f:
    data = json.load(f)["documents"]
    if len(data) > max_visual:
        data = random.sample(data, max_visual)

if __name__ == "__main__":
    ###########
    ## label ##
    ###########

    colors_list = ncolors(6)
    colors_dict = {}
    # for entity in ['other', 'answer', 'question', 'l0']:
    for entity in ['other', 'answer', 'question', 'l1', "l2", "l3"]:
        color = colors_list.pop(0)
        colors_dict[entity] = tuple(color)
    print(colors_dict)

    ########
    ## gt ##
    ########

    def draw_bbox(imgdraw, points, color=''):
        points  = [int(p) for p in points]
        imgdraw.polygon(points, outline=color, width=4)

    font = ImageFont.truetype("exp/han.ttf", 16)

    for label_info in tqdm(data, desc="visual"):
        # img
        img_path    = os.path.join(img_dir, label_info["img"]["fname"])
        img         = Image.open(img_path).convert("RGB")
        img         = ImageOps.exif_transpose(img)
        imgdraw     = ImageDraw.Draw(img)
        # 
        id2center    = {}
        linked_ids   = []
        for doc in label_info["document"]:
            single_box = []
            if len(doc["box"]) == 4:
                x1, y1, x2, y2 = doc["box"]
                single_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                single_box = doc["box"]
                x1, y1, x2, y2 = single_box[0], single_box[1], single_box[4], single_box[5]
            draw_bbox(imgdraw, single_box, color=colors_dict.get(doc["label"], colors_dict['other']))
            id2center[doc["id"]] = ((x1+x2)//2, (y1+y2)//2)
            links = []
            for item in doc["linking"]:
                if len(item) < 3 :
                    links.append((doc["id"], item[0], 0))
                else:
                    links.append(item)
            linked_ids.extend(links)
        # 
        for id1, id2, l in linked_ids:
            x1, y1 = id2center[id1]
            x2, y2 = id2center[id2]
            if 'l'+str(l) in colors_dict:
                imgdraw.line([(x1, y1), (x2, y2)], fill=colors_dict['l'+str(l)], width=6)

        # save
        img.save(os.path.join(save_dir, label_info["img"]["fname"]))

