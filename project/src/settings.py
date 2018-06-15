import json
import os

settings_path = os.path.join(os.getcwd(), "settings.json")

with open(settings_path) as fp:
    data = json.load(fp)


def get_config(item):
    return data[item]
