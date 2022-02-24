from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_meta', type=str, required=True,
                    help="item metadata file")

args = parser.parse_args()
fi = open(args.input_meta, "r")
out_file = args.input_meta.split(".json")[0] + ".csv"
fo = open(out_file, "w")
for line in fi:
    obj = eval(line)
    cat = obj["categories"][0][-1]
    print(obj["asin"] + "\t" + cat, file=fo)
