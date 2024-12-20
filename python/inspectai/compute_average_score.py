import argparse
from collections import defaultdict

from inspect_ai.log import read_eval_log

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str)
    args = parser.parse_args()
    log_file = args.log_file
    
    log = read_eval_log(log_file)
    print("read log")
    scores = defaultdict(list) 
    for sample in log.samples:
        if sample.scores is not None:
            for key, score in sample.scores.items():
                scores[key].append(score.value)

    for key, value in scores.items():
        print(f"Average score for {key}: {sum(value) / len(value)}")