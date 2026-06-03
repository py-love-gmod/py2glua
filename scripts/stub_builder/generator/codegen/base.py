from dataclasses import dataclass


@dataclass
class GeneratedCode:
    code: str
    used_types: set[str]

    @staticmethod
    def empty() -> "GeneratedCode":
        return GeneratedCode("", set())
