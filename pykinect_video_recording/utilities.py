import os

import configparser
import configobj

proceed_messages = ["y", "yes", ""]
redo_messages = ["r", "re", "redo"]


class mouseTrackerException(Exception):
    pass


def set_experiment_and_animal_tag():
    experiment = input("Please enter the experiments name: ")
    tag = input("Please enter the name/tag of the animal: ")
    return experiment, tag


def setup_recording():
    experiment, tag = set_experiment_and_animal_tag()
    proceed_message = input("Experiment: {}, Animal: {}, would you like to proceed? [y]/r/n: " \
                            .format(experiment, tag))
    if proceed_message in proceed_messages:
        return True, experiment, tag
    elif proceed_message in redo_messages:
        setup_recording()
    else:
        return False, experiment, tag


def load_config():
    config = configparser.ConfigParser()
    if "config.cfg" in os.listdir():
        config.read("config.cfg")
        return config
    else:
        raise mouseTrackerException("config.cfg file is missing in {}".format(os.getcwd()))
