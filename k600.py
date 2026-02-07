import os
import csv
from tqdm import tqdm  # 用于显示进度条，如果没有安装可以用 pip install tqdm 安装，或者删除相关代码

def generate_csv(root_dir, output_file):
    print(f"正在遍历目录: {root_dir} ...")
    
    data_rows = []
    video_id = 1
    
    # 设定默认的占位符
    default_action = "unknown_action" 
    default_label = 0

    # 遍历目录
    # os.walk 会递归遍历 root_dir 下的所有文件夹 (vol_00, vol_01...)
    for root, dirs, files in os.walk(root_dir):
        # 过滤出 mp4 文件
        mp4_files = [f for f in files if f.endswith('.mp4')]
        
        # 使用 tqdm 显示当前文件夹的处理进度 (可选)
        for file in tqdm(mp4_files, desc=f"Processing {os.path.basename(root)}", leave=False):
            # 获取文件的绝对路径
            full_path = os.path.join(root, file)
            
            # 这里的 action 我们尝试取父文件夹的名字，通常可能是类别名
            # 如果你的目录结构是 vol_xx/mp4文件，那 action 就会变成 vol_xx
            # 如果你想随便填，可以在下面强制覆盖
            action = os.path.basename(root) 
            
            # 如果你希望 action 完全固定（如用户要求随便填），取消下面这行的注释：
            # action = "test_action"

            data_rows.append([video_id, full_path, action, default_label])
            video_id += 1

    print(f"遍历完成，共找到 {len(data_rows)} 个视频文件。")
    print(f"正在写入 CSV 文件: {output_file} ...")

    # 写入 CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'path', 'action', 'label'])
        # 写入数据
        writer.writerows(data_rows)

    print("完成！")

if __name__ == "__main__":
    # 你的数据根目录
    ROOT_DIR = "/data3/hcyang/avsnet_rebuttal/k600/OpenMMLab___Kinetics600/raw/Kinetics600/videos/"
    # 输出文件名
    OUTPUT_CSV = "k600.csv"
    
    # 检查目录是否存在
    if os.path.exists(ROOT_DIR):
        generate_csv(ROOT_DIR, OUTPUT_CSV)
    else:
        print(f"错误: 找不到目录 {ROOT_DIR}")