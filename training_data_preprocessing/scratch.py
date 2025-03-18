import os 

def counter_files(folder_path):
    """Count the number of files in a folder"""
    return len(os.listdir(folder_path))

print(counter_files("train_old/input"))
print(counter_files("train_old/output"))


