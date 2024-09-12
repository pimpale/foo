import json
from enum import Enum

from Logger import logger


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


# TODO: Add enum for roles
class History:
    def __init__(self):
        self.logs: list[dict[str, str]] = []

    def add(self, role: Role, content: str):
        # anthropic has stricter requirements for the first message, so we need to check that the first message is always the system message
        last_role = None if len(self.logs) == 0 else Role(self.logs[-1]["role"])
        match role, last_role: 
            case Role.system, None:
                pass
            case Role.system, _:
                raise ValueError("Cannot add a system message after other messages have been added.")
            case _, None:
                raise ValueError("Cannot add a message before a system message.")
            case Role.user, Role.user:
                raise ValueError("Cannot add two user messages in a row.")
            case Role.assistant, Role.assistant:
                raise ValueError("Cannot add two assistant messages in a row.")
            case _:
                pass
            
        self.logs.append({"role": role, "content": content})
        self.log_msg(role, content)

    def log_msg(self, role: Role, content: str):
        logger.info(f"===={role.upper()}====")
        logger.info(content + "\n\n\n")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.logs, f)

    def __iter__(self):
        return iter(self.logs)

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, index: int):
        return self.logs[index]

    def __setitem__(self, index: int, value: dict[str, str]):
        self.logs[index] = value
