'''
Rename the Open Images V6 images to adapt pycocotools and produce the corresponding annotations.
The Visual Genome annotations are produced in the same way.
'''
import json
import os

root_path = 'your_path/open-imagev6/' # Download images from Rongjie's repo and unzip it.


with open(root_path +'annotations/'+ 'categories_dict.json') as f:
    categories = json.load(f)
obj_categories = categories['obj']
rel_categories = categories['rel']

categories = []
for idx, i in enumerate(obj_categories):
    category = {'supercategory': i, 'id': idx, 'name': i}
    categories.append(category)

with open(root_path+'annotations/'+'vrd-train-anno.json') as f:
    train_image_list = json.load(f)

counter = 0
new_image_name = 1

images = []
annotations = []
train_rel = {}
#

for row in train_image_list:
    image_path = root_path+'images/'+row['img_fn']+'.jpg' #TODO image_id
    w, h = row['img_size']
    os.rename(image_path, root_path+'images/'+str(new_image_name)+'.jpg')

    image = {'file_name': str(new_image_name)+'.jpg',
              'height': h,
              'width': w,
              'id': new_image_name}
    images.append(image)

    for index, j in enumerate(row['bbox']):

        bbox = [j[0], j[1], j[2]-j[0], j[3]-j[1]] #cxcywh
        area = int(bbox[2] * bbox[3])
        anno_id = counter
        counter = counter + 1

        annotation = {'segmentation': None,
                      'area': area,
                      'bbox': bbox,
                      'iscrowd': 0,
                      'image_id': new_image_name,
                      'id': anno_id,
                      'category_id': row['det_labels'][index]}
        annotations.append(annotation)

    train_rel[new_image_name] = row['rel']

    new_image_name = new_image_name + 1

train_database = {'images': images,
                 'annotations': annotations,
                 'categories': categories}

print('train finish')

with open(root_path+'annotations/'+'vrd-test-anno.json') as f:
    test_image_list = json.load(f)

images = []
annotations = []
test_rel = {}

for row in test_image_list:
    image_path = root_path+'images/'+row['img_fn']+'.jpg' #TODO image_id
    w, h = row['img_size']
    os.rename(image_path, root_path+'images/'+str(new_image_name)+'.jpg')

    image = {'file_name': str(new_image_name)+'.jpg',
              'height': h,
              'width': w,
              'id': new_image_name}
    images.append(image)

    for index, j in enumerate(row['bbox']):

        bbox = [j[0], j[1], j[2]-j[0], j[3]-j[1]] #cxcywh
        area = int(bbox[2] * bbox[3])
        anno_id = counter
        counter = counter + 1

        annotation = {'segmentation': None,
                      'area': area,
                      'bbox': bbox,
                      'iscrowd': 0,
                      'image_id': new_image_name,
                      'id': anno_id,
                      'category_id': row['det_labels'][index]}
        annotations.append(annotation)

    test_rel[new_image_name] = row['rel']

    new_image_name = new_image_name + 1

test_database = {'images': images,
                 'annotations': annotations,
                 'categories': categories}

print('test finish')


with open(root_path+'annotations/'+'vrd-val-anno.json') as f:
    val_image_list = json.load(f)

images = []
annotations = []
val_rel = {}

for row in val_image_list:
    image_path = root_path+'images/'+row['img_fn']+'.jpg' #TODO image_id
    w, h = row['img_size']
    os.rename(image_path, root_path+'images/'+str(new_image_name)+'.jpg')

    image = {'file_name': str(new_image_name)+'.jpg',
              'height': h,
              'width': w,
              'id': new_image_name}
    images.append(image)

    for index, j in enumerate(row['bbox']):

        bbox = [j[0], j[1], j[2]-j[0], j[3]-j[1]] #cxcywh
        area = int(bbox[2] * bbox[3])
        anno_id = counter
        counter = counter + 1

        annotation = {'segmentation': None,
                      'area': area,
                      'bbox': bbox,
                      'iscrowd': 0,
                      'image_id': new_image_name,
                      'id': anno_id,
                      'category_id': row['det_labels'][index]}
        annotations.append(annotation)

    val_rel[new_image_name] = row['rel']

    new_image_name = new_image_name + 1

val_database = {'images': images,
                 'annotations': annotations,
                 'categories': categories}

print('test finish')

rel_database = {'train': train_rel,
                'val': val_rel,
                'test': test_rel,
                'rel_categories': rel_categories}


json.dump(train_database, open('train.json', 'w'))
json.dump(val_database, open('val.json', 'w'))
json.dump(test_database, open('test.json', 'w'))
json.dump(rel_database, open('rel.json', 'w'))
