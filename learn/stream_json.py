import json


def stream_json(filename):
    with open(filename, "r") as file:
        for line in file:
            yield json.loads(line)
