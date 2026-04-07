def get_sytem_prompt(persona):
    text = "get_sytem_prompt"
    return text


def get_sytem_prompt_v2(persona):
    text = "get_sytem_prompt_v2"
    return text


def get_sytem_prompt_v3(persona):
    text = "get_sytem_prompt_v3"

    return text


def get_sytem_prompt_v4(persona):
    text = "get_sytem_prompt_v4"
    return text


def get_basic_dynamic_description_v4(num_tons_lake):
    return "get_basic_dynamic_description_v4"


def get_basic_dynamic_description_v3(num_tons_lake):
    return "get_basic_dynamic_description_v3"


def get_basic_dynamic_description_v2(num_tons_lake):
    return "get_basic_dynamic_description_v2"


def get_basic_dynamic_description(num_tons_lake):
    return "get_basic_dynamic_description"


def question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher):
    return f" If each fisherman catches {num_tons_fisher} tons, how many tons of fish will there be next month?"


def question_shrinking_limit(num_tons_lake):
    return f" What is the maximum number of tons that each fisherman can catch without shrinking the lake population, i.e. we still have {num_tons_lake} tons of fish next month?"


def question_shrinking_limit_assumption(num_tons_lake):
    return f"Assume each fisherman catches the same amount of fish. What is the maximum number of tons each fisherman can catch so that after the population doubles, there will be at least {num_tons_lake} tons of fish?"


def prompt_description_simulation(persona_name, num_tons_lake):
    return f"""\nLocation: lake
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before everyone fishes, there are {num_tons_lake} tons of fish in the lake.
"""


import math


def prompt_description_simulation_univ(persona_name, num_tons_lake):
    overfishing_level = (num_tons_lake // 2) // 5
    return f"""\nLocation: lake
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before everyone fishes, there are {num_tons_lake} tons of fish in the lake.
- 2024-01-01: Given the current situation, if everyone fishes more than {overfishing_level} tons, the lake population will shrink next month.
"""
