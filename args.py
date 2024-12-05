import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from myfunctions import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--data_path', type=str,
                    default="./dataset/data/",
                    help='The directory containing the parking lot data.')
parser.add_argument('--Atten_path', type=str,
                    default="./dataset/atten/",
                    help='The directory containing the similar attention matrixs.')
parser.add_argument('--temp_path', type=str,
                    default="./dataset/temperature_data.csv",
                    help='The path of the temperature data.')
parser.add_argument('--LOOK_BACK', type=int, default=12,
                    help='Number of time step of the Look Back Mechanism.')
parser.add_argument('--predict_time', type=int, default=12,
                    help='Number of time step of the predict time.')
parser.add_argument('--nodes', type=int, default=58,
                    help='Number of parking lots.')
parser.add_argument('--meta_epochs', type=int, default=300,
                    help='The meta training epochs.')
parser.add_argument('--finetuning_epochs', type=int, default=600,
                    help='The fine tuning training epochs.')
parser.add_argument('--inner_lr', type=float, default=0.01,
                    help='The learning rate of of the support set.')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='The learning rate of of the query set.')
parser.add_argument('--meta_rate', type=float, default=0.5,
                    help='The rate of meta training.')
parser.add_argument('--fine_tuning_rate', type=float, default=0.2,
                    help='The rate of fine tuning training.')
parser.add_argument('--no_rain_list', type=list, default=[3, 9, 10, 11, 13, 14, 15, 17, 19, 20, 24, 26, 27, 28, 29],
                    help='The list of no rain days.')
parser.add_argument('--light_rain_list', type=list, default=[12, 18, 21, 25, 30],
                    help='The list of light rain days.')
parser.add_argument('--heavy_rain_list', type=list, default=[1, 2, 4, 5, 6, 7, 8, 16, 22, 23],
                    help='The list of heavy rain days.')
parser.add_argument('--k', type=int, default=3,
                    help='Number of weather condition.')
parser.add_argument('--seq_len', type=int, default=12,
                    help='Number of time step of the input sequence.')
parser.add_argument('--MLP_hidden', type=int, default=64,
                    help='Hidden size of MLP.')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Alpha.')
parser.add_argument('--layer', type=float, default=4,
                    help='number of HALSTM input feature.')


args = parser.parse_args(args=[]) # jupyter
# args = parser.parse_args()      # python

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(2023)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
