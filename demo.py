import simpy
from simpy.resources.resource import Release, Request

class LinkResource(simpy.Resource):
    def __init__(self, env: simpy.Environment, capacity: int = 1):
        super().__init__(env, capacity)
        self.cnt = 0
    
    def request(self) -> Request:
        self.cnt += 1
        if self.cnt > 1:
            return False
        return super().request()

    def release(self, request: Request) -> Release:
        
        return super().release(request)
    