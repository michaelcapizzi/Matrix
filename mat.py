coursera = 1
# Copyright 2013 Philip N. Klein
from vec import Vec


#Test your Mat class over R and also over GF(2).  The following tests use only R.

def getitem(M, k):
    """
    Returns the value of entry k in M, where k is a 2-tuple
    >>> M = Mat(({1,3,5}, {'a'}), {(1,'a'):4, (5,'a'): 2})
    >>> M[1,'a']
    4
    >>> M[3,'a']
    0
    """
    assert k[0] in M.D[0] and k[1] in M.D[1]

    return M.f.get((k[0], k[1]), 0)

def equal(A, B):
    """
    Returns true iff A is equal to B.

    Consider using brackets notation A[...] and B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> Mat(({'a','b'}, {'A','B'}), {('a','B'):0}) == Mat(({'a','b'}, {'A','B'}), {('b','B'):0})
    True
    >>> A = Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1})
    >>> B = Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1, ('b','B'):0})
    >>> C = Mat(({'a','b'}, {'A','B'}), {('a',1):2, ('b','A'):1, ('b','B'):5})
    >>> A == B
    True
    >>> B == A
    True
    >>> A == C
    False
    >>> C == A
    False
    >>> A == Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1})
    True
    """
    assert A.D == B.D

    #make A dense
    for i in A.D[0]:
        for j in A.D[1]:
            if (i,j) not in A.f.keys():
                A.f[(i,j)] = 0
            else:
                continue

    #make B dense
    for i in B.D[0]:
        for j in B.D[1]:
            if (i,j) not in B.f.keys():
                B.f[(i,j)] = 0
            else:
                continue

    return True if A.f == B.f else False

def setitem(M, k, val):
    """
    Set entry k of Mat M to val, where k is a 2-tuple.
    >>> M = Mat(({'a','b','c'}, {5}), {('a', 5):3, ('b', 5):7})
    >>> M['b', 5] = 9
    >>> M['c', 5] = 13
    >>> M == Mat(({'a','b','c'}, {5}), {('a', 5):3, ('b', 5):9, ('c',5):13})
    True

    Make sure your operations work with bizarre and unordered keys.

    >>> N = Mat(({((),), 7}, {True, False}), {})
    >>> N[(7, False)] = 1
    >>> N[(((),), True)] = 2
    >>> N == Mat(({((),), 7}, {True, False}), {(7,False):1, (((),), True):2})
    True
    """
    assert k[0] in M.D[0] and k[1] in M.D[1]

    M.f[k] = val

def add(A, B):
    """
    Return the sum of Mats A and B.

    Consider using brackets notation A[...] or B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> A1 = Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (6,'y'):3})
    >>> A2 = Mat(({3, 6}, {'x','y'}), {(3,'y'):4})
    >>> B = Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (3,'y'):4, (6,'y'):3})
    >>> A1 + A2 == B
    True
    >>> A2 + A1 == B
    True
    >>> A1 == Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (6,'y'):3})
    True
    >>> zero = Mat(({3,6}, {'x','y'}), {})
    >>> B + zero == B
    True
    >>> C1 = Mat(({1,3}, {2,4}), {(1,2):2, (3,4):3})
    >>> C2 = Mat(({1,3}, {2,4}), {(1,4):1, (1,2):4})
    >>> D = Mat(({1,3}, {2,4}), {(1,2):6, (1,4):1, (3,4):3})
    >>> C1 + C2 == D
    True
    """
    assert A.D == B.D

    new_dict = {}

    for i in A.D[0]:
        for j in A.D[1]:
            new_dict[(i,j)] = A.f.get((i,j),0) + B.f.get((i,j), 0)

    return Mat(A.D, new_dict)


def scalar_mul(M, x):
    """
    Returns the result of scaling M by x.

    >>> M = Mat(({1,3,5}, {2,4}), {(1,2):4, (5,4):2, (3,4):3})
    >>> 0*M == Mat(({1, 3, 5}, {2, 4}), {})
    True
    >>> 1*M == M
    True
    >>> 0.25*M == Mat(({1,3,5}, {2,4}), {(1,2):1.0, (5,4):0.5, (3,4):0.75})
    True
    """

    #if scalar is 0 then return empty dictionary
    if x == 0:
        new_dict = {}
    #else compute scalar multiplication
    else:
        new_dict = {}

        for i in M.D[0]:
            for j in M.D[1]:
                new_dict[(i,j)] = x * M[(i,j)]

    return Mat(M.D, new_dict)


def transpose(M):
    """
    Returns the matrix that is the transpose of M.

    >>> M = Mat(({0,1}, {0,1}), {(0,1):3, (1,0):2, (1,1):4})
    >>> M.transpose() == Mat(({0,1}, {0,1}), {(0,1):2, (1,0):3, (1,1):4})
    True
    >>> M = Mat(({'x','y','z'}, {2,4}), {('x',4):3, ('x',2):2, ('y',4):4, ('z',4):5})
    >>> Mt = Mat(({2,4}, {'x','y','z'}), {(4,'x'):3, (2,'x'):2, (4,'y'):4, (4,'z'):5})
    >>> M.transpose() == Mt
    True
    """

    new_dict = {}
    new_domain = (M.D[1], M.D[0])

    for i in M.f.keys():
        if M.f[i] != 0:
            new_dict[(i[1], i[0])] = M.f[i]
    # for i in M.D[0]:
    #     for j in M.D[1]:
    #         if (i,j) in M.f.keys() and M.f[(i,j)] != 0:
    #             new_dict[(j,i)] = M.f[(i,j)]
        else:
            continue

    return Mat(new_domain, new_dict)

#TODO - more sparsity can come from adjusting vec.add
def vector_matrix_mul(v, M):
    """
    returns the product of vector v and matrix M

    Consider using brackets notation v[...] in your procedure
    to access entries of the input vector.  This avoids some sparsity bugs.

    >>> v1 = Vec({1, 2, 3}, {1: 1, 2: 8})
    >>> M1 = Mat(({1, 2, 3}, {'a', 'b', 'c'}), {(1, 'b'): 2, (2, 'a'):-1, (3, 'a'): 1, (3, 'c'): 7})
    >>> v1*M1 == Vec({'a', 'b', 'c'},{'a': -8, 'b': 2, 'c': 0})
    True
    >>> v1 == Vec({1, 2, 3}, {1: 1, 2: 8})
    True
    >>> M1 == Mat(({1, 2, 3}, {'a', 'b', 'c'}), {(1, 'b'): 2, (2, 'a'):-1, (3, 'a'): 1, (3, 'c'): 7})
    True
    >>> v2 = Vec({'a','b'}, {})
    >>> M2 = Mat(({'a','b'}, {0, 2, 4, 6, 7}), {})
    >>> v2*M2 == Vec({0, 2, 4, 6, 7},{})
    True
    >>> v3 = Vec({'a','b'},{'a':1,'b':1})
    >>> M3 = Mat(({'a', 'b'}, {0, 1}), {('a', 1): 1, ('b', 1): 1, ('a', 0): 1, ('b', 0): 1})
    >>> v3*M3 == Vec({0, 1},{0: 2, 1: 2})
    True
    """
    assert M.D[0] == v.D

    if M.f == {} or v.f == {} or set(v.f.values()) == {0}:
        return Vec(M.D[1], {})
    else:

        #make list of column vectors
        veclist = [Vec(M.D[1], {md1: M.f[(vd, md1)] * v.f[vd] for md1 in M.D[1] if (vd, md1) in M.f.keys() and M.f[(vd, md1)] != 0}) for vd in v.f.keys() if vd in v.f.keys() and v.f[vd] != 0]

        #sum vectors in veclist
        accum = veclist[0]

        for i in range(1, len(veclist)):        #for i 1 to len(veclist)
            accum = accum + veclist[i]

        #remove 0s for sparsity
        new_dict = {}
        for k in accum.f.keys():
            if accum.f[k] != 0:
                new_dict[k] = accum.f[k]
            else:
                continue

        #return accum                   #if not needed to remove 0s for sparsity
        return Vec(M.D[1], new_dict)

#TODO - more sparsity can come from adjusting vec.add
def matrix_vector_mul(M, v):
    """
    Returns the product of matrix M and vector v.

    Consider using brackets notation v[...] in your procedure
    to access entries of the input vector.  This avoids some sparsity bugs.

    >>> N1 = Mat(({1, 3, 5, 7}, {'a', 'b'}), {(1, 'a'): -1, (1, 'b'): 2, (3, 'a'): 1, (3, 'b'):4, (7, 'a'): 3, (5, 'b'):-1})
    >>> u1 = Vec({'a', 'b'}, {'a': 1, 'b': 2})
    >>> N1*u1 == Vec({1, 3, 5, 7},{1: 3, 3: 9, 5: -2, 7: 3})
    True
    >>> N1 == Mat(({1, 3, 5, 7}, {'a', 'b'}), {(1, 'a'): -1, (1, 'b'): 2, (3, 'a'): 1, (3, 'b'):4, (7, 'a'): 3, (5, 'b'):-1})
    True
    >>> u1 == Vec({'a', 'b'}, {'a': 1, 'b': 2})
    True
    >>> N2 = Mat(({('a', 'b'), ('c', 'd')}, {1, 2, 3, 5, 8}), {})
    >>> u2 = Vec({1, 2, 3, 5, 8}, {})
    >>> N2*u2 == Vec({('a', 'b'), ('c', 'd')},{})
    True
    >>> M3 = Mat(({0,1},{'a','b'}),{(0,'a'):1, (0,'b'):1, (1,'a'):1, (1,'b'):1})
    >>> v3 = Vec({'a','b'},{'a':1,'b':1})
    >>> M3*v3 == Vec({0, 1},{0: 2, 1: 2})
    True
    """
    assert M.D[1] == v.D

    if M.f == {} or v.f == {} or set(v.f.values()) == {0}:
        return Vec(M.D[0], {})
    else:

        #make list of column vectors
        veclist = [Vec(M.D[0], {md0: M.f[(md0, vd)] * v.f[vd] for md0 in M.D[0] if (md0, vd) in M.f.keys() and M.f[(md0, vd)] != 0}) for vd in v.f.keys() if vd in v.f.keys() and v.f[vd] != 0]
        # veclist = [Vec(M.D[0], {md0: M.f[(md0, vd)] * v.f[vd] for (md0,vd) in M.f.keys() if M.f[(md0, vd)] != 0}) for vd in v.f.keys() if vd in v.f.keys() and v.f[vd] != 0]

        #sum vectors in veclist
        accum = veclist[0]

        for i in range(1, len(veclist)):        #for i 1 to len(veclist)
            accum = accum + veclist[i]

        #remove 0s for sparsity
        new_dict = {}
        for k in accum.f.keys():
            if accum.f[k] != 0:
                new_dict[k] = accum.f[k]
            else:
                continue

        #return accum                   #if not needed to remove 0s for sparsity
        return Vec(M.D[0], new_dict)


#TODO - add sparsity implementation
#TODO - more sparsity can come from adjusting vec.add
def matrix_matrix_mul(A, B):
    """
    Returns the result of the matrix-matrix multiplication, A*B.

    Consider using brackets notation A[...] and B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> A = Mat(({0,1,2}, {0,1,2}), {(1,1):4, (0,0):0, (1,2):1, (1,0):5, (0,1):3, (0,2):2})
    >>> B = Mat(({0,1,2}, {0,1,2}), {(1,0):5, (2,1):3, (1,1):2, (2,0):0, (0,0):1, (0,1):4})
    >>> A*B == Mat(({0,1,2}, {0,1,2}), {(0,0):15, (0,1):12, (1,0):25, (1,1):31})
    True
    >>> C = Mat(({0,1,2}, {'a','b'}), {(0,'a'):4, (0,'b'):-3, (1,'a'):1, (2,'a'):1, (2,'b'):-2})
    >>> D = Mat(({'a','b'}, {'x','y'}), {('a','x'):3, ('a','y'):-2, ('b','x'):4, ('b','y'):-1})
    >>> C*D == Mat(({0,1,2}, {'x','y'}), {(0,'y'):-5, (1,'x'):3, (1,'y'):-2, (2,'x'):-5})
    True
    >>> M = Mat(({0, 1}, {'a', 'c', 'b'}), {})
    >>> N = Mat(({'a', 'c', 'b'}, {(1, 1), (2, 2)}), {})
    >>> M*N == Mat(({0,1}, {(1,1), (2,2)}), {})
    True
    >>> E = Mat(({'a','b'},{'A','B'}), {('a','A'):1,('a','B'):2,('b','A'):3,('b','B'):4})
    >>> F = Mat(({'A','B'},{'c','d'}),{('A','d'):5})
    >>> E*F == Mat(({'a', 'b'}, {'d', 'c'}), {('b', 'd'): 15, ('a', 'd'): 5})
    True
    >>> F.transpose()*E.transpose() == Mat(({'d', 'c'}, {'a', 'b'}), {('d', 'b'): 15, ('d', 'a'): 5})
    True
    """
    assert A.D[1] == B.D[0]

    #make a vector of each matrix row
        #from matutil
    #{row:Vec(A.D[1], {col:A[row,col] for col in A.D[1]}) for row in A.D[0]}
    #treat it like multiple vector_matrix_mul problems where each row of matrix is a vector
    #DONE ==> convert list of vector-rows into matrix
    #Mat(({0,1}, a.D), {(i,j): ab[i].f[j] for i in {0,1} for j in a.D})

    rowdomain = A.D[0]
    columndomain = B.D[1]

    if A.f == {} or B.f == {}:
        return Mat((rowdomain, columndomain), {})
    else:
        mat2vecsdict = {row:Vec(A.D[1], {col:A[row,col] for col in A.D[1] if A[row,col] != 0}) for row in A.D[0]}
        mat2vecs = [v for v in mat2vecsdict.values()]       #does this always maintain the order of rows?
        mat = [v * B for v in mat2vecs]
        return Mat((rowdomain, columndomain), {(list(rowdomain)[i],j): mat[i].f[j] for i in range(len(rowdomain)) for j in columndomain if j in mat[i].f.keys()})







################################################################################

class Mat:
    def __init__(self, labels, function):
        assert isinstance(labels, tuple)
        assert isinstance(labels[0], set) and isinstance(labels[0], set)
        assert isinstance(function, dict)
        self.D = labels
        self.f = function

    __getitem__ = getitem
    __setitem__ = setitem
    transpose = transpose

    def __neg__(self):
        return (-1)*self

    def __mul__(self,other):
        if Mat == type(other):
            return matrix_matrix_mul(self,other)
        elif Vec == type(other):
            return matrix_vector_mul(self,other)
        else:
            return scalar_mul(self,other)
            #this will only be used if other is scalar (or not-supported). mat and vec both have __mul__ implemented

    def __rmul__(self, other):
        if Vec == type(other):
            return vector_matrix_mul(other, self)
        else:  # Assume scalar
            return scalar_mul(self, other)

    __add__ = add

    def __radd__(self, other):
        "Hack to allow sum(...) to work with matrices"
        if other == 0:
            return self

    def __sub__(a,b):
        return a+(-b)

    __eq__ = equal

    def copy(self):
        return Mat(self.D, self.f.copy())

    def __str__(M, rows=None, cols=None):
        "string representation for print()"
        if rows == None: rows = sorted(M.D[0], key=repr)
        if cols == None: cols = sorted(M.D[1], key=repr)
        separator = ' | '
        numdec = 3
        pre = 1+max([len(str(r)) for r in rows])
        colw = {col:(1+max([len(str(col))] + [len('{0:.{1}G}'.format(M[row,col],numdec)) if isinstance(M[row,col], int) or isinstance(M[row,col], float) else len(str(M[row,col])) for row in rows])) for col in cols}
        s1 = ' '*(1+ pre + len(separator))
        s2 = ''.join(['{0:>{1}}'.format(str(c),colw[c]) for c in cols])
        s3 = ' '*(pre+len(separator)) + '-'*(sum(list(colw.values())) + 1)
        s4 = ''.join(['{0:>{1}} {2}'.format(str(r), pre,separator)+''.join(['{0:>{1}.{2}G}'.format(M[r,c],colw[c],numdec) if isinstance(M[r,c], int) or isinstance(M[r,c], float) else '{0:>{1}}'.format(M[r,c], colw[c]) for c in cols])+'\n' for r in rows])
        return '\n' + s1 + s2 + '\n' + s3 + '\n' + s4

    def pp(self, rows, cols):
        print(self.__str__(rows, cols))

    def __repr__(self):
        "evaluatable representation"
        return "Mat(" + str(self.D) +", " + str(self.f) + ")"

    def __iter__(self):
        raise TypeError('%r object is not iterable' % self.__class__.__name__)
