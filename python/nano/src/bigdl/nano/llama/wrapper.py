import argparse
import os
import subprocess


def llama(args):
    exec_path = 'llama.cpp/main'
    command = f'''{exec_path} \
        -t {args.threads} \
        -p {args.prompt} \
        -n {args.n_predict} \
        -m {args.model}
    '''
    subprocess.run(command.split(), check=True)


def bloomz(args):
    exec_path = 'bloomz.cpp/main'
    command = f'''{exec_path} \
        -t {args.threads} \
        -p {args.prompt} \
        -n {args.n_predict} \
        -m {args.model}
    '''
    subprocess.run(command.split(), check=True)


def redpajama(args):
    exec_path = 'redpajama.cpp/main'
    command = f'''{exec_path} \
        -t {args.threads} \
        -p {args.prompt} \
        -n {args.n_predict} \
        -m {args.model}
    '''
    subprocess.run(command.split(), check=True)


model_exec_map = {
    'llama': llama,
    'bloomz': bloomz,
    'redpajama': redpajama
}


def get_model_family(args):
    # TODO: get model family by model path content
    global model_exec_map
    model_basename = os.path.basename(args.model)
    for model_family in model_exec_map.keys():
        if model_family in model_basename:
            return model_family
    model_dirname = os.path.basename(os.path.dirname(args.model))
    for model_family in model_exec_map.keys():
        if model_family in model_dirname:
            return model_family
    return 'llama'


def main():
    parser = argparse.ArgumentParser(description='Command line wrapper')

    parser.add_argument('-t', '--threads', type=int, default=28,
                        help='number of threads to use during computation (default: 28)')
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='prompt to start generation with (default: empty)')
    parser.add_argument('-n', '--n_predict', type=int, default=-1,
                        help='number of tokens to predict (default: -1, -1 = infinity)')
    parser.add_argument('-m', '--model', type=str,
                        default='llama.cpp/models/lamma-7B/ggml-model.bin',
                        help='model path (default: llama.cpp/models/lamma-7B/ggml-model.bin)')
    parser.add_argument('--model_family', type=str, default='',
                        choices=['llama', 'bloomz', 'redpajama'],
                        help='family name of model')

    args = parser.parse_args()

    # Print the values of the parsed arguments
    print('Threads:', args.threads)
    print('Prompt:', args.prompt)
    print('N Predict:', args.n_predict)
    print('Model:', args.model)

    if args.model_family:
        model_exec_map[args.model_family](args)
    else:
        model_exec_map[get_model_family(args)](args)


if __name__ == '__main__':
    main()
