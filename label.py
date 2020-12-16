import numpy as np


class Label:
    def __init__(self, points):
        tmp = np.asarray(points).reshape((-1,))
        self.x1 = tmp[0]
        self.y1 = tmp[1]
        self.x2 = tmp[2]
        self.y2 = tmp[3]

    def __str__(self):
        return "({x1},{y1}),({x2},{y2})".format(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)

    def pascal_voc(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def pascal_voc_with_name(self):
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2), "somelabel")

    def display_form(self):
        return ((self.x1, self.y1),
                (self.x1, self.y2),
                (self.x2, self.y2),
                (self.x2, self.y1))

    def shift(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy
        return self

    def scale(self, cx, cy):
        self.x1 *= cx
        self.x2 *= cx
        self.y1 *= cy
        self.y2 *= cy
        return self


def to_display_labels(labels):
    return np.asarray(list(Label(label).display_form() for label in labels))