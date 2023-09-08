# How Robust is Federated Learning to Communication Error? A Comparison Study between Uplink and Downlink
# For MNIST experiments:
cd mnist  
if checking how robust to downlink BER:  
    python usersx_berx_dw.py --iid --users 5 --dnber 1e-1 (users and dnber can be changed)  
if checking how robust to uplink BER:  
    uplink model updates: python usersx_berx_dw.py --iid --users 5 --upber 1e-1 (users and upber can be changed)  
    uplink model weights: python usersx_berx_w.py --iid --users 5 --upber 1e-1 (users and upber can be changed)  
# For Fashion-MNIST experiments:
cd fashion_mnist  
if checking how robust to downlink BER:  
    python usersx_berx_dw.py --iid --users 5 --dnber 1e-1 (users and dnber can be changed)  
if checking how robust to uplink BER:  
    uplink model updates: python usersx_berx_dw.py --iid --users 5 --upber 1e-1 (users and upber can be changed)  
    uplink model weights: python usersx_berx_w.py --iid --users 5 --upber 1e-1 (users and upber can be changed)  
