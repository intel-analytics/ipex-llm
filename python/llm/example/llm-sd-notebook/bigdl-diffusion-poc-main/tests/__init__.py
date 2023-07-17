import os

TEST_ROOT = os.path.dirname(os.path.realpath(__file__))
RESOURCES_FOLDER = os.path.join(TEST_ROOT, 'resources')
OUTPUTS_FOLDER = os.path.join(TEST_ROOT, 'outputs')

def get_resource(name):
    return os.path.join(RESOURCES_FOLDER, name)

def get_output(name):
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)
    return os.path.join(OUTPUTS_FOLDER, name)