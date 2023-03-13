#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from . import TRIGGER_REG_NAME_PREFIX
import time


def triggerbyfile(filename):
    '''
    File trigger, will be activated once the file is modified.

    :param filename: the path to the file to be modified.
    '''
    from inotify_simple import INotify, flags

    def wrapped_func(func):
        def new_func(*args, **kwargs):
            while True:
                inotify = INotify()
                watch_flags = flags.MODIFY
                wd = inotify.add_watch(filename, watch_flags)
                for event in inotify.read():
                    func(*args, **kwargs)
        new_func.__name__ = TRIGGER_REG_NAME_PREFIX + func.__name__
        return new_func
    return wrapped_func


def triggerbyclock(seconds=1):
    '''
    Clock trigger, will be activated every fixed seconds

    :param seconds: the time interval(in seconds) to activate the trigger.
    '''
    def wrapped_func(func):
        def new_func(*args, **kwargs):
            while True:
                time.sleep(seconds)
                func(*args, **kwargs)
        new_func.__name__ = TRIGGER_REG_NAME_PREFIX + func.__name__
        return new_func
    return wrapped_func

# TODO: more triggers to be added
