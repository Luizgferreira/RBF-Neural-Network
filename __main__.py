import argparse
import os
import sys

from models.models import RBFN
from test.cross_validation import CV
from training.training import train

PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)

def get_arguments():
    parser = argparse.ArgumentParser(add_help=False, prog="testarg")
    parser.add_argument('--input', required=False, nargs='?', help='input file')
    parser.add_argument('--train', required=False, const=True, nargs='?', help='train the model')
    parser.add_argument('--test', required=False, const=True , nargs='?', help='test the model using cross-validation')
    args = parser.parse_args()
    return args

#print(main_path)
def run(args):
    if args.input:
        model = RBFN()
        model.loadData()
        model.readPredict(os.path.join(PATH,args.input))
    elif args.train:
        model = RBFN()
        model.train()
    elif args.test:
        print(CV())

def begin():
    args = get_arguments()
    run(args)

if __name__ == '__main__':
    ar = begin()
