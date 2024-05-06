import tkinter as tk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Get Rectangle Coordinates")

        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()

        # Create the rectangle on the canvas
        self.rect = self.canvas.create_line(100, 100, 200, 200, width=2)

        # Button to get rectangle coordinates
        self.button = tk.Button(root, text="Get Rectangle Coordinates", command=self.get_rect_coords)
        self.button.pack()

    def get_rect_coords(self):
        # Get the coordinates of the rectangle
        rect_coords = self.canvas.coords(self.rect)
        print("Rectangle Coordinates:", rect_coords)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
