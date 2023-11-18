import sys

# x0 interval start
# x1 interval end
# a BF index 0-1
# x point in interval

# Evaluate BF at point
# a must be 0 or 1 or it errors out
def NBasis(a,x0,x1,x):
    denom = x1-x0
    if a == 0:
        numer = x1-x
    elif a == 1:
        numer = x-x0
    else:
        sys.exit("Basis function index must be either 0 or 1")
    return numer/denom

def NBasisDerv(a,x0,x1,x):
    denom = x1-x0
    if a == 0:
        numer = -1
    elif a == 1:
        numer = 1
    else:
        sys.exit("Basis function index must be either 0 or 1")
    return numer/denom

# x0 destination interval start
# x1 destination interval end
# xi point in domain -1 1

# Evaluate XMap at point xi (map xi to a point x in [x0, x1])
def XMap(x0,x1,xi):
    if -1 <= xi <= 1:
        x = 0
    else:
       sys.exit("xi must be between -1 and 1") 
    xvals = [x0,x1]
    for a in range(0,2):
        x += NBasis(a,-1,1,xi) * xvals[a]
    return x

def XMapDerv(x0,x1,xi):
    if -1 <= xi <= 1:
        x_derv = 0
    else:
       sys.exit("xi must be between -1 and 1") 
    xvals = [x0,x1]
    for a in range(0,2):
        x_derv += NBasisDerv(a,-1,1,xi) * xvals[a]
    return x_derv