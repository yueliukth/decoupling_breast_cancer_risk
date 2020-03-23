import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# train
os.system("python paper2196/train_network.py 'network1-512-imagenet-lr2-newmean-b32-standardization-randomcolor-sigmoid-asgd-leaky-seed1' 'continue_latest_50_100'")

os.system("python paper2196/train_network.py 'network2-512-imagenet-lr1-newmean-b32-standardization-randomcolor-sigmoid-leaky-asgd-dropout-seed1' 'continue_latest_20_50'")

os.system("python paper2196/train_network.py 'network3-512-imagenet-lr2-newmean-b32-standardization-randomcolor-sigmoid-leaky-asgd-seed1' 'continue_latest_20_50'")

# test
os.system("python paper2196/eval_network.py 'network1-512-imagenet-lr2-newmean-b32-standardization-randomcolor-sigmoid-asgd-leaky-seed71' 'test_test_3264'")

os.system("python paper2196/eval_network.py 'network2-512-imagenet-lr1-newmean-b32-standardization-randomcolor-sigmoid-leaky-asgd-dropout-seed1' 'test_test_13696'")

os.system("python paper2196/eval_network.py 'network3-512-imagenet-lr2-newmean-b32-standardization-randomcolor-sigmoid-leaky-asgd-seed1' 'test_test_11799'")

