from tensorflow.contrib.framework.python.framework import checkpoint_utils
import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output.',
                        type=str,
                        required=False)
    return parser
parser = get_arg_parser()
args = parser.parse_args()
checkpoint_dir = args.checkpoint
for name, shape in checkpoint_utils.list_variables(checkpoint_dir):
     print('loading...', name, shape, checkpoint_utils.load_variable(checkpoint_dir,name))