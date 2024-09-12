import subprocess

from commands.BaseCommand import BaseCommand, CallbackType
from commands.CommandBlocker import CommandBlocker


class BashCommand(BaseCommand):
    def __init__(self, timeout: int, callback: CallbackType | None = None):
        self.timeout = timeout
        super().__init__(
            xml_tag="bash",
            description="""To run a shell command, wrap it in <bash></bash> XML tags. Examples:
<bash>ls</bash>
<bash>python3 script.py</bash>
<bash>cat file.txt</bash>
<bash>python3 -c "
import numpy as np
print(np.sum([1, 2, 3]))
"</bash>""",
            callback=callback,
        )

    def _run(self, content: str) -> str:
        if CommandBlocker.should_block(content):
            return "BASH ERROR:\nInteractive command not allowed."

        try:
            result = subprocess.run(
                content, shell=True, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return "BASH ERROR:\nCommand timed out."

        output = []

        if result.stdout:
            output.append(f"BASH OUTPUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"BASH ERROR:\n{result.stderr}")

        if not output:
            return "BASH OUTPUT:\nCommand ran successfully with no output."

        return "\n".join(output)
