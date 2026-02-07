import os
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
import hashlib

def extract_image_bytes(img_data):
    """处理多种可能的图像数据格式"""
    if isinstance(img_data, bytes):
        return img_data
    elif isinstance(img_data, dict):
        # 尝试从字典中提取字节数据
        if 'bytes' in img_data:
            return img_data['bytes']
        elif 'data' in img_data:
            return img_data['data']
    elif hasattr(img_data, 'bytes'):
        return img_data.bytes
    raise ValueError(f"无法识别的图像数据格式: {type(img_data)}")

def parquet_to_folders(parquet_dir, output_dir):
    """转换Parquet文件到标准文件夹结构"""
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    
    for file in tqdm(parquet_files, desc="Processing files"):
        file_path = os.path.join(parquet_dir, file)
        try:
            # 使用更高效的低内存读取方式
            pf = pq.ParquetFile(file_path)
            
            # 通过batch迭代减少内存使用
            for batch in pf.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                
                # 确定数据集类型
                if 'train' in file:
                    subset = 'train'
                elif 'val' in file:
                    subset = 'val'
                elif 'test' in file:
                    subset = 'test'
                else:
                    continue
                
                for _, row in df.iterrows():
                    try:
                        label = str(row.get("label", "unknown"))
                        img_data = row["image"]
                        
                        # 转换图像数据
                        img_bytes = extract_image_bytes(img_data)
                        
                        # 生成唯一文件名
                        if 'id' in row:
                            img_filename = f"{row['id']}.JPEG"
                        elif 'name' in row:
                            img_filename = row['name']
                        else:
                            # 使用图像内容的哈希作为文件名
                            img_filename = f"{hashlib.md5(img_bytes).hexdigest()}.JPEG"
                        
                        # 保存图像
                        class_dir = os.path.join(output_dir, subset, label)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        with Image.open(io.BytesIO(img_bytes)) as img:
                            img.save(os.path.join(class_dir, img_filename))
                            
                    except Exception as e:
                        print(f"\n处理文件 {file} 的行时出错: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"\n处理文件 {file} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    parquet_dir = "/data2/YHCDYP/imagenet/dir/data"
    output_dir = "/data2/YHCDYP/imagenet/imagenet_standard"
    
    print("开始转换...")
    parquet_to_folders(parquet_dir, output_dir)
    print(f"转换完成！结果保存在 {output_dir}")