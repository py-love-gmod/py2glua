from .gen_class import generate as gen_class


def generate(name: str, data: dict) -> str:
    return gen_class(name, data)
