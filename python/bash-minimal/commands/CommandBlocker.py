class CommandBlocker:
    """Blocks interactive commands to prevent infinite stalls."""

    blocklist = ["vim", "vi", "emacs", "nano", "nohup", "git"]
    blocklist_standalone = [
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "exit",
        "/bin/bash",
        "/bin/sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
    ]

    @staticmethod
    def should_block(action: str) -> bool:
        args = action.strip().split()
        if len(args) == 0:
            return False

        command = args[0]
        if command in CommandBlocker.blocklist:
            return True

        is_standalone = len(args) == 1
        if is_standalone and command in CommandBlocker.blocklist_standalone:
            return True

        return False
