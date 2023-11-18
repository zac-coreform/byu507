import sys

class BoundaryCondition():
    def __init__(self, bc_posn, bc_type, b1, b2, b3):
        self.bc_type = bc_type
        self.bc_posn = bc_posn        
        if self.bc_posn == "left":
            self.bc_index = 0
        elif self.bc_posn == "right":
            self.bc_index = 1
        else:
            sys.exit("Unknown BC position: must be left or right")
        
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.isDir = False
        self.isNeu = False
        self.isRob = False
        
        if self.bc_type == "Dir":
            self.isDir = True
            self.g_val = self.b3 / self.b1
        elif self.bc_type == "Neu":
            self.isNeu = True
            self.h_val = self.b3 / self.b2
        elif self.bc_type == "Rob":
            self.isRob = True
            self.r_val = self.b3 / self.b2
            self.r_val_k = self.b1 / self.b2
        else:
            sys.exit("Unknown BC type: must be Dir, Neu, or Rob")
