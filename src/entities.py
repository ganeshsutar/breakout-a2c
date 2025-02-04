from collections import namedtuple

Experience = namedtuple(
    "Experience", ["state", "next_state", "action", "reward", "done"]
)

Stat = namedtuple("Stat", ["episode", "reward", "length"])
