import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rounds', type=int, default=50, help="rounds of training") 
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C") 
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--upbit', type=int, default=8, help="node transmit bit length")
    parser.add_argument('--upber', type=float, default=1e-8, help="ldpc bit error rate")
    parser.add_argument('--dnbit', type=int, default=8, help="server broadcast bit length")
    parser.add_argument('--dnber', type=float, default=1e-8, help="ldpc bit error rate")

    parser.add_argument('--model', type=str, default='vanillacnn', help='model name')
    parser.add_argument('--dataset', type=str, default='fashionmnist', help="name of dataset") 
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")  
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    args = parser.parse_args()
    return args
    
   
