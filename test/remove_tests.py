import os
import shutil

SMALL_TEST_DIR = "./smalltest_networkx"
LARGE_TEST_DIR = "./largetest_networkx"

def clear_test_files(test_dir):
    for filename in os.listdir(test_dir):
        file_path = os.path.join(test_dir, filename)
        if os.path.isdir(file_path):
            for sub_filename in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, sub_filename)
                if os.path.isdir(sub_file_path):
                    shutil.rmtree(sub_file_path) 
                elif os.path.isfile(sub_file_path): 
                    os.remove(sub_file_path)
        elif os.path.isfile(file_path): 
            os.remove(file_path)

clear_test_files(SMALL_TEST_DIR)
clear_test_files(LARGE_TEST_DIR)
