import re
from abc import ABC, abstractmethod
from collections.abc import Callable

CallbackType = Callable[[], None]


class BaseCommand(ABC):
    def __init__(self, xml_tag: str, description: str, callback: CallbackType | None = None):
        self.xml_tag = xml_tag
        self.description = description
        self.callback = callback

    def count_occurrences(self, response: str) -> int:
        return len(self.extract_content(response))

    def extract_content(self, response: str) -> list[str]:
        """Extracts content from innermost XML-like tags in the response string."""
        pattern = re.compile(
            f"<{self.xml_tag}>((?:(?!<{self.xml_tag}>).)*?)</{self.xml_tag}>", re.DOTALL
        )
        return pattern.findall(response)

    @abstractmethod
    def _run(self, content: str) -> str:
        pass

    def execute(self, response: str) -> list[str]:
        results = []
        contents = self.extract_content(response)
        for content in contents:
            result = self._run(content)
            if self.callback:
                self.callback()
            results.append(result)
        return results

    def __str__(self):
        return f"{self.xml_tag}: {self.description}"
