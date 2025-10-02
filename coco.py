import os
import zipfile

# 路径设置
zip_dir = '/local_datasets/coco2017zip/'              # 你的压缩包所在路径
target_dir = '/local_datasets/coco2017/' # 解压后存放标准结构的位置

# 创建目标目录结构
os.makedirs(os.path.join(target_dir, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'train2017'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'val2017'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'test2017'), exist_ok=True)

# 映射每个 zip 文件到目标路径
extract_map = {
    'annotations_trainval2017.zip': 'annotations',
    'train2017.zip': 'train2017',
    'val2017.zip': 'val2017',
    'test2017.zip': 'test2017'
}

# 解压函数
def unzip_to_folder(zip_path, output_folder):
    print(f'Extracting {zip_path} to {output_folder}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

# 解压每个文件到标准位置
for zip_file, target_subdir in extract_map.items():
    zip_path = os.path.join(zip_dir, zip_file)
    out_path = os.path.join(target_dir, target_subdir)

    # special case: annotations_trainval2017.zip 解压后文件很多，要移动进去
    if zip_file.startswith('annotations'):
        unzip_to_folder(zip_path, os.path.join(target_dir, 'annotations'))
    else:
        unzip_to_folder(zip_path, target_dir)  # 直接解压到根
        # 如果内容解压后是 train2017/xxx.jpg，已在根下，不需要额外移动

print("✅ 解压完成，已组织为标准 COCO2017 文件结构。")

# nohup python tools/train.py configs/faster_rcnn/train_faster-rcnn_r50_coco2017.py > train.log 2>&1 &

