from commands.BaseCommand import BaseCommand, CallbackType


class SubmitCommand(BaseCommand):
    def __init__(self, submission_path: str, callback: CallbackType | None = None):
        super().__init__(
            xml_tag="submit",
            description="To submit an answer, wrap it in <submit></submit> XML tags. Example: <submit>YOUR ANSWER HERE</submit>",
            callback=callback,
        )
        self.submission_path = submission_path

    def save_file(self, file_path: str, content: str):
        with open(file_path, "w") as f:
            f.write(content)

    def _run(self, content: str) -> str:
        content = content.strip()

        self.save_file(self.submission_path, content)

        return "Successfully submitted answer."
