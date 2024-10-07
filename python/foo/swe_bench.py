from typing import cast
import datasets
import subprocess
from swe_bench_constants import SpecInstance, SWEbenchInstance
from test_spec import get_test_specs_from_dataset, make_test_spec

# in order to speed up evaluation, we only include 10 swebench tasks.
TASK_SUBSET = ["django__django-11039"]

CACHE_DIR = "/root/swebench_cache/"

class TaskFamily:
    """
    Example milestone task.
    """

    standard_version = "0.2.3"

    @staticmethod
    def install():
        # save dataset to disk
        dataset = datasets.load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
        if not isinstance(dataset, datasets.Dataset):
            raise ValueError("Dataset is not of type datasets.Dataset")
        dataset.save_to_disk("/root/swebench.hf")
        
        # cache all the repos
        test_specs = get_test_specs_from_dataset(CACHE_DIR, [cast(SWEbenchInstance, d) for d in dataset])
        
        # for each test spec, execute the steps
        for spec in test_specs:
            if spec.instance_id in TASK_SUBSET:
                subprocess.run(spec.setup_env_script, shell=True, executable="/bin/bash")
                subprocess.run(spec.install_repo_script, shell=True, executable="/bin/bash")
        
        # copy the conda environment to the agent
        subprocess.run("cp -r /root/.conda /home/agent/.conda", shell=True)


    @staticmethod
    def get_instructions(t: SWEbenchInstance) -> str:
        return f"""
Please fix the bug described below:
```
{t['problem_statement']}
```
Notes:
The repository is located at: ~/{t['instance_id']}
There is a corresponding conda environment ({t['instance_id']}) that is activated by default. You shouldn't need to install dependencies.
You may need to pip install -e . the repo before running the tests.
"""

    @staticmethod
    def get_tasks() -> dict[str, SWEbenchInstance]:
        dataset = cast(list[SWEbenchInstance], datasets.load_from_disk("/root/swebench.hf"))
        result = {}
        for item in dataset:
            result[item['instance_id']] = item
        return result

    @staticmethod
    def start(t: SWEbenchInstance):
        # copy the repo to the agent
        subprocess.run(f"cp -r /root/swebench_cache/{t['instance_id']} /home/agent/{t['instance_id']}", shell=True)
        # add the conda activate to the .bashrc
        subprocess.run(f"echo 'source /opt/miniconda3/bin/activate {t['instance_id']}' >> /home/agent/.bashrc", shell=True)

    @staticmethod
    def score(t: SWEbenchInstance, submission: str) -> int:
        # apply the patch
        t['patch'] = submission
        spec = make_test_spec(CACHE_DIR, t)
        subprocess.run(spec.eval_script, shell=True, executable="/bin/bash")
        return 0


    @staticmethod
    def get_permissions(t: SWEbenchInstance) -> list[str]:
        return ["full_internet"]
    