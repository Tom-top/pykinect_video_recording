import os

import configobj
from validate import Validator

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
    config_file_name = "config.cfg"
    config_spec_file_name = "config.configspec"
    if config_file_name in os.listdir() and config_spec_file_name in os.listdir():
        config = configobj.ConfigObj(config_file_name, configspec=config_spec_file_name)
        validator = Validator()
        valid = config.validate(validator)
        if valid:
            return config
        else:
            raise mouseTrackerException("config could not ba validated")
    else:
        raise mouseTrackerException("config.cfg file is missing in {}".format(os.getcwd()))
