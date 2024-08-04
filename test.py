import os  

def rename_files_in_directory(directory):  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            # 获取文件扩展名  
            name, ext = os.path.splitext(file)  
            if ext in ['.json', '.png']:  
                try:  
                    # 将文件名转换为三位数，不足部分前面补零  
                    new_name = f'{int(name):03d}{ext}'  
                    old_path = os.path.join(root, file)  
                    new_path = os.path.join(root, new_name)  
                    
                    # 重命名文件  
                    os.rename(old_path, new_path)  
                    print(f'Renamed: {old_path} --> {new_path}')  
                except ValueError:  
                    print(f'Skipping file: {file} (not a number)')  

if __name__ == '__main__':  
    directory = 'dataset/train'  # 请将此路径替换为你的目标目录路径  
    rename_files_in_directory(directory)