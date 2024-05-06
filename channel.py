
class Channel(object):
    def __init__(self, num_links: int):
        self.channels_tx_sta = num_links * [[]]
    
        