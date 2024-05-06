"""Drawing editor main file"""
import tkinter as tk
from tkinter import ttk
from idlelib.tooltip import Hovertip
from rectangle import Rectangle
from line import Line
from selector import Selector

args = {
    'rectangle': ['./assets/rectangle.png', 'Rectangle Button:\nThis is used to'
                  'draw a rectangle'],
    'line': ['./assets/line.png', 'Line Button:\nThis is used to draw a line']
}

class App(tk.Tk):
    """Window properties"""
    def __init__(self):
        super().__init__()
        self.attributes('-fullscreen', True)
        self.title("Pentagon: Drawing Editor")
        self.minsize(800,600)
        self.config(cursor="arrow")

        # initialize sidebar
        self.sidebar = Sidebar(self)

        # initialize a drawing canvas
        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.num_buttons = 0

        # create rectangle button
        self.rectangle_button = self.__create_button__(Rectangle(self).callback,
                                                   args['rectangle'])

        # create a line button
        self.line_button = self.__create_button__(Line(self).callback,
                                                   args['line'])

        # set default selections
        self.selector = Selector(self)
        self.init_bindings(self.selector)

        # object list: contains a list of objects
        self.currlist = []
        self.mainloop()

    def __create_button__(self, callback, arguments):
        """Function to add a button"""
        image_path, tooltip = arguments
        button = ttk.Button(self.sidebar,
                            image=tk.PhotoImage(file=image_path),
                            command=callback, cursor="hand2")

        # add button to sidebar and dictionary
        index = self.num_buttons
        button.grid(row=index//2,column=index%2, padx=5, pady=5)
        self.num_buttons += 1

        Hovertip(button, tooltip)
        return button

    def init_bindings(self, selector):
        """Initializes default selections"""
        self.canvas.bind("<Button-1>", selector.perform_selection)
        self.canvas.bind("<Control-Button-1>", selector.activate_multi)
        self.canvas.bind("<Button-3>", selector.perform_deselection)



class Sidebar(ttk.Frame): # pylint: disable=too-many-ancestors
    """Class for intilizing sidebar"""
    def __init__(self, parent):
        super().__init__(parent, width=200)
        self.pack(side=tk.LEFT, fill=tk.Y)

if __name__=='__main__':
    App()
