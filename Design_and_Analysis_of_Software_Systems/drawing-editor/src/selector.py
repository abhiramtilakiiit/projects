"""Used for selecting objects"""

class Selector():
    """Object use for selection logic"""
    def __init__(self, parent):
        self.selected_objects = set()
        self.root = parent
        self.canvas = parent.canvas
        self.multimode = False

    def perform_selection(self):
        """add a current object to selector"""
        curr_x = self.canvas.winfo_pointerx()
        curr_y = self.canvas.winfo_pointery()
        for obj in self.root.currlist:
            item_type = self.canvas.type(obj)
            if item_type == "rectangle":
                o_x, o_y 
                
            if o_x - 1 < curr_x <= o_x+1 and o_y - 1 < curr_y <= o_y+1:
                self.selected_objects.add(obj)



    def perform_deselection(self):
        if not self.multimode:
            self.selected_objects = set()
        

    def activate_multi(self, event):
        """activating the multiselect mode"""
        i
        self.multimode = True

    def deactivate_multi(self, event):
        """deactivating the multiselect mode"""
        self.multimode = False
