import glob
import json
import matplotlib.pyplot as plt
import os

def visualize_1000():
    json_file_list = glob.glob("./result/*")
    
    plt.figure(figsize=(10, 10))
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            data = json.load(f)
        name = os.path.basename(json_file).split(".")[0]
        train_loss = data['train_loss'][:9]
        test_loss = data['test_loss'][:9]
        if '20' in name:
            color = 'red'
        elif '32' in name:
            color = 'purple'
        elif '44' in name:
            color ='blue'
        elif '56' in name:
            color = 'green'
        elif '110' in name:
            color ='black'
    
        if "True" in name:
            line_style = '-'
        else:
            line_style='--'
        plt.plot(range(1, len(train_loss) + 1), train_loss, linestyle=line_style, label=name, color=color)
        plt.ylabel("Train Loss")
        plt.xlabel("iteration(10e-2)")
        plt.legend()
    save_path = "./graph/effect_of_residual_mapping_in_1000iter.png"    
    # Save
    plt.savefig(save_path)
    print(f"Saved in {save_path}")

def visualize_64000(target):
    json_file_list = glob.glob("./result/*")
    plt.figure(figsize=(10, 10))
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            data = json.load(f)
        name = os.path.basename(json_file).split(".")[0]
        target_list = data[target]
        if '20' in name:
            color = 'red'
        elif '32' in name:
            color = 'purple'
            continue
        elif '44' in name:
            color ='blue'
            continue
        elif '56' in name:
            color = 'green'
        elif '110' in name:
            color ='black'
            continue
    
        if "True" in name:
            line_style = '-'
        else:
            line_style='--'
        plt.plot(range(1, len(target_list) + 1), target_list, linestyle=line_style, label=name, color=color)
        plt.ylabel(target)
        plt.xlabel("iteration(10e-2)")        
        if target == 'train_loss':
            min_ylim = 0
            max_ylim = 0.5
        elif target == 'test_loss':
            min_ylim = 0.2
            max_ylim = 1
        elif target == 'train_acc':
            min_ylim = 0.8
            max_ylim = 1.001
        elif target == 'test_acc':
            min_ylim = 0.8
            max_ylim = 0.9
        plt.ylim(min_ylim, max_ylim)
        plt.legend()
    save_path = f"./graph/{target}_in_64000iter.png"
    # Save
    plt.savefig(save_path)
    print(f"Saved in {save_path}")

if __name__ == "__main__":
    target_list = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    # visualize_1000()
    # for target in target_list:
    #     visualize_64000(target)
    
