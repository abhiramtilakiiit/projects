"""Used for selecting objects"""

class Selector():
    """Object use for selection logic"""
    def __init__(self, parent):
        self.selected_objects = set()
        self.root = parent
        self.canvas = parent.canvas
        self.rounded_circles = {}

    def perform_selection(self, event, multi=False):
        """add a current object to selector"""
        if not multi:
            self.selected_objects = set()
        for obj in self.root.currlist:
            print(obj)
            obj_type = obj and self.canvas.type(obj)
            if obj_type == "rectangle" and self.__rect_intersection__(event, obj):
                if obj in self.selected_objects:
                    self.selected_objects.remove(obj)
                    self.__reflect_selected__(obj, True, False)
                else:
                    self.selected_objects.add(obj)
                    self.__reflect_selected__(obj, True)
                break
            if obj_type == "line" and self.__line_intersection__(event, obj):
                if obj in self.selected_objects:
                    self.selected_objects.remove(obj)
                    self.__reflect_selected__(obj, False, False)
                else:
                    self.selected_objects.add(obj)
                    self.__reflect_selected__(obj, False)
                break

    def perform_deselection(self, _event):
        """used to remove all currently selected elements"""
        for obj in self.selected_objects:
            obj_type = obj and self.canvas.type(obj)
            self.__reflect_selected__(obj, not obj_type == 'line', False)

        self.selected_objects = set()
        self.rounded_circles = set()


    def activate_multi(self, event):
        """activating the multiselect mode"""
        self.perform_selection(event, True)

    def __distance_from_line__(self, x_x, y_y, first, second):
        c1_x, c1_y = first
        c2_x, c2_y = second

        # Calculate the components of the line vector
        d_x = c2_x - c1_x
        d_y = c2_y - c1_y

        segment_length_squared = d_x**2 + d_y**2
        scale = ((x_x - c1_x) * d_x + (y_y - c1_y) * d_y) / segment_length_squared
        closest_point_on_line = (c1_x + scale * d_x, c1_y + scale * d_y)
        distance = (x_x - closest_point_on_line[0])**2 + (y_y - closest_point_on_line[1])**2

        return distance

    def __rect_intersection__(self, event, obj):
        x_x, y_y = event.x, event.y
        c1_x, c1_y, c3_x, c3_y = self.canvas.coords(obj)
        first = c1_x, c1_y
        third = c3_x, c3_y
        second = c1_x, c3_y
        fourth = c3_x, c1_y
        points_list = [first, second, third, fourth]

        # distance from line
        for i in range(4):
            if self.__distance_from_line__(x_x, y_y, points_list[i],
                                           points_list[(i+1)%4]) < 5:
                return True
        return False

    def __line_intersection__(self, event, obj):
        x_x, y_y = event.x, event.y
        c1_x, c1_y, c2_x, c2_y = self.canvas.coords(obj)
        if self.__distance_from_line__(x_x, y_y, (c1_x, c1_y), (c2_x, c2_y)) < 5:
            return True
        return False

    def __reflect_selected__(self, obj, outline, start=True):
        color = 'lightblue' if start else 'black'
        if outline:
            self.canvas.itemconfig(obj, outline=color)
        else:
            self.canvas.itemconfig(obj, fill=color)

        # make circles at ends
        c1_x, c1_y, c2_x, c2_y = self.canvas.coords(obj)
        radius = 3

        o1_x = c1_x - radius
        o1_y = c1_y - radius
        o2_x = c1_x + radius
        o2_y = c1_y + radius
        oval_1 = self.canvas.create_oval(o1_x, o1_y, o2_x, o2_y)

        o1_x = c2_x - radius
        o1_y = c2_y - radius
        o2_x = c2_x + radius
        o2_y = c2_y + radius
        oval_2 = self.canvas.create_oval(o1_x, o1_y, o2_x, o2_y)

        self.rounded_circles.update({obj: (oval_1, oval_2)})
