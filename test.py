import json
import glob

for i in range(150):
    dataset_path = f'dataset/test/*/{str(i)}.json'
    dirs = glob.glob(dataset_path)
    with open(f'dataset/car/{str(i)}.json','r') as f:
        new_label = json.load(f)
    for dir in dirs:
        with open(dir,'r') as f:
            print(dir)
            old_label = json.load(f)
            old_label['rpoints'] = new_label['rpoints']
        with open(dir,'w') as f:
            json.dump(old_label, f)


    