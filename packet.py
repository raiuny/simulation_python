
class Pkt(object):
    def __init__(self, id, arr_time = -1, ser_time = -1, dep_time = -1, num = -1):
        self.id = id
        self.num = num
        self.arr_time = arr_time
        self.ser_time = ser_time # 成为HOL包的时间
        self.dep_time = dep_time # 成功传输的时间