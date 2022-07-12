# class to represent the data obtained from the yaml files
# needed taking in consideration the format of the data for better readability
# each entity represents a rectangle obtained from the cards image
# cx, cy is the center of the rectangle, width and height are the size, angle the inclination and rotation
# and the rectangle attribute represents the matrix of pixels that are found at the specific center of the rectangle
class CardData:
    def __init__(self, cx, cy, width, height, angle, card_id, rectangle):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.angle = angle
        self.card_id = card_id
        self.rectangle = rectangle

    def __str__(self):
        print("cx: ", self.cx, " cy: ", self.cy, " width: ", self.width, " height: ", self.height,
              " angle: ", self.angle, " card_id: ", self.card_id)