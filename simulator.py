from simpy import Environment
class Simulator(object):
    def __init__(self, env: Environment):
        self.env = env
    
    def run(self):
        self.env.run()