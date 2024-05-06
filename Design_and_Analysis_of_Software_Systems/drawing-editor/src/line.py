"""Implement line drawing"""

class Line():
	"""Button with line that is used to draw"""
	def __init__(self, parent):
		self.draw_mode = False
		self.root = parent
		self.canvas = parent.canvas

		# to be used in drawing line
		self.current_line = None
		self.coords = None

	def callback(self):
		"""call when line is clicked"""
		self.draw_mode = not self.draw_mode
		if self.draw_mode:
			self.__change_cursor__()
			self.canvas.bind('<Button-1>', self.start_line)
			self.canvas.bind('<B1-Motion>', self.draw_line)
			self.canvas.bind('<ButtonRelease-1>', self.end_line)
			self.canvas.bind('<Button-3>', self.exit_drawing)
		else:
			self.exit_drawing(None)

	def __change_cursor__(self):
		current_cursor = self.root.cget("cursor")
		if current_cursor == "arrow":
			self.root.config(cursor="tcross")
		else:
			self.root.config(cursor="arrow")

	def start_line(self, event):
		"""initialize a line"""
		self.coords = (event.x, event.y, event.x, event.y)

	def draw_line(self, event):
		"""dynamic line drawer"""
		if self.current_line:
			self.canvas.delete(self.current_line)

		x_1, y_1, x_2, y_2 = self.coords
		x_2, y_2 = event.x, event.y
		self.current_line = self.canvas.create_line(x_1, y_1, x_2, y_2,
													fill='black')

	def end_line(self, _event):
		"""end drawing this line"""
		self.root.currlist.append(self.current_line)
		self.current_line = None
		self.coords = None

	def exit_drawing(self, _event):
		"""exit drawing"""

		# unbind
		self.__change_cursor__()
		self.canvas.unbind('<Button-1>')
		self.canvas.unbind('<B1-Motion>')
		self.canvas.unbind('<ButtonRelease-1>')
		self.canvas.unbind('<Button-3>')
		self.root.init_bindings(self.root.selector)

		self.current_line = None
		self.coords = None
		self.draw_mode = False
