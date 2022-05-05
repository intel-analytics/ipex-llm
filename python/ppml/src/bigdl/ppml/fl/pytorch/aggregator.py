

class Aggregator(object):
    def __init__(self) -> None:
        self.model = None
        self.client_data = {}
        
    def add_server_model(self, model):
        if self.model is not None:
            raise Exception("model already exists on server")
        self.model = model

    def aggregate(self):


