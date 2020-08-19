import argparse
import re


def main(args):
    results = []
    with open(args.file_path, "r") as f:
        epoch = -1
        reading = False
        for line in f:
            line = line.strip()
            if line.startswith("TRAIN: Epoch:"):
                epoch = int(re.search("[0-9]+", line).group())
                reading = True
            if reading and line.startswith("OVERALL (Micro)"):
                f1 = float(line.split("\t")[-1])
                results.append((f1, epoch))
                reading = False
    sorted_results = sorted(results, reverse=True)[:5]
    for entry in sorted_results:
        print("Epoch: {0} | Overall Micro F1: {1}".format(entry[1], entry[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Parse Result Files")
    ap.add_argument("--file_path", type=str, default="/Users/jarora/Desktop/results/config_results/experiment.tsv",
                    help="path of results file to parse")
    ap = ap.parse_args()
    main(ap)
