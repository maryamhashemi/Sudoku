class Sudoko_Solver:
    
    def __init__(self):
        self.digits   = '123456789'
        self.rows     = 'ABCDEFGHI'
        self.cols     = self.digits
        self.squares  = self.cross(self.rows, self.cols)
        self.unitlist = ([self.cross(self.rows, c) for c in self.cols] +
                    [self.cross(r, self.cols) for r in self.rows] +
                    [self.cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
        
        self.units = dict((s, [u for u in self.unitlist if s in u]) 
                     for s in self.squares)
        
        self.peers = dict((s, set(sum(self.units[s],[]))-set([s]))
                     for s in self.squares)
            
        self.test()
        
    '''
        Cross product of elements in A and elements in B.
    '''
    def cross(self,A, B):
        
        return [a+b for a in A for b in B]

    def test(self):
        "A set of unit tests."
        assert len(self.squares) == 81
        assert len(self.unitlist) == 27
        assert all(len(self.units[s]) == 3 for s in self.squares)
        assert all(len(self.peers[s]) == 20 for s in self.squares)
        assert self.units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                               ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                               ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
        assert self.peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                                   'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                                   'A1', 'A3', 'B1', 'B3'])
        print ('All tests pass.')

    '''
        Convert grid to a dict of possible values, {square: digits}, or
        return False if a contradiction is detected.
    '''
    def parse_grid(self,grid):
        # To start, every square can be any digit; then assign values from the grid.
        values = dict((s, self.digits) for s in self.squares)
        for s,d in self.grid_values(grid).items():
            if d in self.digits and not self.assign(values, s, d):
                return False ## (Fail if we can't assign d to square s.)
        return values

    '''
        Convert grid into a dict of {square: char} with '0' or '.' for empties.
    '''
    def grid_values(self, grid): 
        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))

    '''
        Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return False if a contradiction is detected.
    '''
    def assign(self, values, s, d):
        other_values = values[s].replace(d, '')
        if all(self.eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return False

    '''Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected.'''
    def eliminate(self, values, s, d):
        if d not in values[s]:
            return values ## Already eliminated
        values[s] = values[s].replace(d,'')
        ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
        if len(values[s]) == 0:
            return False ## Contradiction: removed last value
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self.eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False
        ## (2) If a unit u is reduced to only one place for a value d, then put it there.
        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False ## Contradiction: no place for this value
            elif len(dplaces) == 1:
                # d can only be in one place in unit; assign it there
                    if not self.assign(values, dplaces[0], d):
                        return False
        return values

    '''
        Display these values as a 2-D grid.
    '''
    def display(self,values):
        width = 1+max(len(values[s]) for s in self.squares)
        line = '+'.join(['-'*(width*3)]*3)
        for r in self.rows:
            print (''.join(values[r+c].center(width)+('|' if c in '36' else '')
                          for c in self.cols))
            if r in 'CF': 
                print (line)
        print()  

    def solve(self,grid): 
        grid = self.convertGridtoString(grid)
        self.display(self.grid_values(grid))
        grid = self.search(self.parse_grid(grid))
        if(grid == False):
            print("Not Solvable")
            return False
        else :
            self.display(grid)
            return grid
        
    '''
        Using depth-first search and propagation, try all possible values.
    '''
    def search(self,values):
        
        if values is False:
            return False ## Failed earlier
        if all(len(values[s]) == 1 for s in self.squares): 
            return values ## Solved!
        ## Chose the unfilled square s with the fewest possibilities
        n,s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        return self.some(self.search(self.assign(values.copy(), s, d)) 
            for d in values[s])

    '''
        Return some element of seq that is true.
    '''
    def some(self,seq):
       
        for e in seq:
            if e: return e
        return False   


    def convertGridtoString(self,grid):
        
        strgrid = ''
        for num in grid:
            strgrid += str(num)
            
        return strgrid
            



