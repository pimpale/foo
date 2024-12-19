import argparse
from typing import Optional
from inspect_ai.log import EvalLog, EvalSample, list_eval_logs, read_eval_log, write_eval_log

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    log_dir = args.log_dir
    output = args.output
    
    # the base log we use to construct the output log
    log: Optional[EvalLog] = None
    
    # the samples we take
    samples: dict[str, EvalSample] = {}
    
    for log_file in list_eval_logs(log_dir):
        print(f"Reading log file: {log_file}")
        
        log = read_eval_log(log_file)
        if log.samples is None:
            print(f"Log file {log_file} has no samples")
            continue
        for sample in log.samples:
            if sample.scores is not None:
                print(f"Sample {sample.id} has {len(sample.scores)} scores")
                samples[sample.id] = sample
    
    # construct the output log
    if log is not None:
        print("Writing output log")
        log.samples = list(samples.values())
        write_eval_log(log, output)        
    else:
        print("No logs found")
