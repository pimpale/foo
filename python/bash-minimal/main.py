import json
from dataclasses import dataclass

from Agent import Agent, AgentArguments
from commands import BashCommand, SubmitCommand, BrowseCommand
from Logger import logger
from simple_parsing import parse
from simple_parsing.helpers import FlattenedAccess, FrozenSerializable
from templates import DEMONSTRATION, DEMONSTRATION_TEMPLATE, INSTRUCTION_TEMPLATE, SYSTEM_TEMPLATE


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    agent: AgentArguments
    instructions: str  # The instructions for the agent to follow.
    submission_path: str = "/home/agent/submission.txt"  # The path to save the submission to.
    show_demonstration: bool = True  # Whether to show the demonstration.


def initialize_agent(args: ScriptArguments):
    agent = Agent(args.agent)
    commands = [
        SubmitCommand(args.submission_path, agent._submit_callback),
        BashCommand(args.agent.bash_timeout),
        BrowseCommand(),
    ]
    command_descriptions = "\n".join([str(command) for command in commands])
    agent.add_commands(commands)
    sys_msg = SYSTEM_TEMPLATE.format(command_descriptions=command_descriptions)
    if args.show_demonstration:
        sys_msg += DEMONSTRATION_TEMPLATE.format(demonstration=DEMONSTRATION)
    agent.add_system_msg(sys_msg)
    agent.add_user_msg(INSTRUCTION_TEMPLATE.format(instructions=args.instructions))
    return agent


def log_args(args: ScriptArguments):
    args_dict = args.to_dict()
    del args_dict["agent"]["model"]["openai_api_key"]
    del args_dict["agent"]["model"]["together_api_key"]
    del args_dict["agent"]["model"]["fireworks_api_key"]
    del args_dict["agent"]["model"]["anthropic_api_key"]
    arg_str = json.dumps(args_dict, indent=4)

    logger.info(f"====RUN ARGS====\n{arg_str}\n\n\n")


def main(args: ScriptArguments):
    try:
        log_args(args)
        agent = initialize_agent(args)
        agent.loop()
        agent.save_history("history.json")
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt caught", exc_info=True)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main(parse(ScriptArguments))
