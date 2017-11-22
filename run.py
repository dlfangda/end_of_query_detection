import argparse
from main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path')
    parser.add_argument('--mfb_dim',default=40)
    parser.add_argument('--lstm_width',default=64)
    parser.add_argument('--dense_layer_width',default=64)
    parser.add_argument('--output_dim',default=2)
    parser.add_argument('--nb_conv_filters',1)
    parser.add_argument('--filter_length',8)
    opts = parser.parse_args()
    main(opts)