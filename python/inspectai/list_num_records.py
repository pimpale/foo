import argparse
from inspect_ai.log import list_eval_logs, read_eval_log

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str)
    args = parser.parse_args()
    log_dir = args.log_dir
    
    resume_ids = set()
    for log_file in list_eval_logs(log_dir):
        print(f"Reading log file: {log_file}")
        
        log = read_eval_log(log_file)
        if log.samples is None:
            print(f"Log file {log_file} has no samples")
            continue
        for sample in log.samples:
            if sample.scores is not None:
                print(f"Sample {sample.id} has {len(sample.scores)} scores")
                resume_ids.add(sample.id)
                
    print(f"Found {len(resume_ids)} samples with scores")