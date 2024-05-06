"""Button function implementations in python"""

class Rectangle():
	"""Button with rectangle that is used to draw"""
	def __init__(self, parent):
		self.draw_mode = False
		self.root = parent
		self.canvas = parent.canvas

		# to be used in drawing rectangle
		self.current_rect = None
		self.coords = None

	def callback(self):
		"""call when rectangle is clicked"""
		self.draw_mode = not self.draw_mode
		if self.draw_mode:
			self.__change_cursor__()
			self.canvas.bind('<Button-1>', self.start_rect)
			self.canvas.bind('<B1-Motion>', self.draw_rect)
			self.canvas.bind('<ButtonRelease-1>', self.end_rect)
			self.canvas.bind('<Button-3>', self.exit_drawing)
		else:
			self.exit_drawing(None)

	def __change_cursor__(self):
		current_cursor = self.root.cget("cursor")
		if current_cursor == "arrow":
			self.root.config(cursor="tcross")
		else:
			self.root.config(cursor="arrow")

	def start_rect(self, event):
		"""initialize a rectangle"""
		self.coords = (event.x, event.y, event.x, event.y)

	def draw_rect(self, event):
		"""dynamic rectangle drawer"""
		if self.current_rect:
			self.canvas.delete(self.current_rect)

		x_1, y_1, x_2, y_2 = self.coords
		x_2, y_2 = event.x, event.y
		self.current_rect = self.canvas.create_rectangle(x_1, y_1, x_2, y_2,
														 outline='black')

	def end_rect(self, _event):
		"""end drawing this rectangle"""
		self.root.currlist.append(self.current_rect)
		self.current_rect = None
		self.coords = None

	def exit_drawing(self, _event):
		"""exit drawing"""

		# unbind
		self.__change_cursor__()
		self.canvas.unbind('<Button-1>')
		self.canvas.unbind('<B1-Motion>')
		self.canvas.unbind('<ButtonRelease-1>')
		self.canvas.unbind('<Button-3>')

		self.current_rect = None
		self.coords = None
		self.draw_mode = False
