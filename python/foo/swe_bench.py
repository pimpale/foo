import datasets
import subprocess
from swe_bench_constants import SpecInstance, SWEbenchInstance
from test_spec import get_test_specs_from_dataset

# in order to speed up evaluation, we only include 10 swebench tasks.
TASK_SUBSET = []

class TaskFamily:
    """
    Example milestone task.
    """

    standard_version = "0.2.3"

    @staticmethod
    def install():
        # save dataset to disk
        dataset = datasets.load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
        dataset.save_to_disk("/root/swebench.hf")
        
        # cache all the repos
        test_specs = get_test_specs_from_dataset("/root/swebench_cache/", dataset)
        
        # for each test spec, execute the steps
        for spec in test_specs:
            if spec.instance_id in TASK_SUBSET:
                for command in spec.env_script_list:
                    subprocess.run(command, shell=True)
                for command in spec.repo_script_list:
                    subprocess.run(command, shell=True)
        
        # copy the conda environment to the agent
        subprocess.run("cp -r /root/.conda /home/agent/.conda", shell=True)


    @staticmethod
    def get_instructions(t: SWEbenchInstance) -> str:
        question = t['question']
        formatted_choices = "\n".join([f"{i}. {choice}" for i, choice in enumerate(t['choices'])])
        return f"""Please answer the question below. Return only your answer, which should be the number of the correct choice.\n{question}\nChoices:\n{formatted_choices}"""

    @staticmethod
    def get_tasks() -> dict[str, SWEbenchInstance]:
        dataset = datasets.load_from_disk("/root/swebench.hf")
        result = {}
        for item in dataset:
            result[item['instance_id']] = item
        return result

    @staticmethod
    def start(t: SWEbenchInstance) -> SpecInstance:
        # copy the repo to the agent
        subprocess.run(f"cp -r /root/swebench_cache/{t['instance_id']} /home/agent/{t['instance_id']}", shell=True)

    @staticmethod
    def score(t: SWEbenchInstance, submission: str) -> int:
        return int(submission.strip() == str(t["answer"]))

    @staticmethod
    def get_permissions(t: SWEbenchInstance) -> list[str]:
        return ["full_internet"]
    
    