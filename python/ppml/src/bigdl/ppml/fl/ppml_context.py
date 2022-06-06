from bigdl.ppml.fl import *


class PPMLContext(JavaValue):
    def __init__(self, jvalue=None, *args):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, *args)


if __name__ == '__main__':
    ppml_context = PPMLContext()
