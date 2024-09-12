from dataclasses import dataclass

from APIModel import ModelArguments, get_model
from commands import BaseCommand
from History import History, Role
from templates import NO_COMMANDS_CALLED, TOO_MANY_COMMANDS


@dataclass(frozen=True)
class AgentArguments:
    model: ModelArguments
    message_cap: int = 30  # The maximum number of messages the agent can send.
    bash_timeout: int = 20  # The maximum seconds a bash command can run for.


class Agent:
    def __init__(self, args: AgentArguments):
        self.history = History()
        self.message_left = args.message_cap
        self.model = get_model(args.model)
        self.commands: list[BaseCommand] = []
        self.has_submitted = False

    def loop(self):
        while not self.has_submitted and self.message_left > 0:
            response = self.model.query(self.history)
            self.history.add(role=Role.assistant, content=response)
            self._handle_commands(response)
            self.message_left -= 1

    def add_commands(self, commands: list[BaseCommand]):
        self.commands.extend(commands)

    def add_system_msg(self, content: str):
        self.history.add(role=Role.system, content=content)

    def add_user_msg(self, content: str):
        self.history.add(role=Role.user, content=content)

    def save_history(self, path: str):
        self.history.save(path)

    def _total_command_calls(self, content: str) -> int:
        return sum(command.count_occurrences(content) for command in self.commands)

    def _handle_commands(self, content: str):
        total_call_num = self._total_command_calls(content)
        if total_call_num == 0:
            return self.history.add(role=Role.user, content=NO_COMMANDS_CALLED)
        if total_call_num > 1:
            return self.history.add(role=Role.user, content=TOO_MANY_COMMANDS)

        self._execute_commands(content)

    def _execute_commands(self, content: str):
        for command in self.commands:
            outputs = command.execute(content)

            for output in outputs:
                self.history.add(role=Role.user, content=output)

    def _submit_callback(self):
        self.has_submitted = True
