class VEC2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    @property
    def re(self):
        return self.x
    @re.setter
    def re(self, value):
        self.x = value
    @property
    def im(self):
        return self.y
    @im.setter
    def im(self, value):
        self.y = value

class VEC2I:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    @property
    def px(self):
        return self.x
    @px.setter
    def px(self, value):
        self.x = value
    @property
    def pa(self):
        return self.y
    @pa.setter
    def pa(self, value):
        self.y = value
    @property
    def nx(self):
        return self.x
    @nx.setter
    def nx(self, value):
        self.x = value
    @property
    def ny(self):
        return self.y
    @ny.setter
    def ny(self, value):
        self.y = value