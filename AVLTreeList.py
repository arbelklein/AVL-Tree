# username - maibendayan
# id1      - 319071148
# name1    - Mai Ben Dayan
# id2      - 208278945
# name2    - Arbel Klein


"""
    A class representing a node in an AVL tree
"""
import random


class AVLNode(object):
    """
        Constructor for leaf node.

        @type value: str
        @param value: data of your node

        Time Complexity - O(1)
    """

    def __init__(self, value=None):
        if value is None:  # constructing virtual node.
            self.value = None
            self.left = None
            self.right = None
            self.parent = None
            self.height = -1  # height of node. BF uses this field. height of virtual node (== empty node) is -1.
            self.isVirtual = True  # field to see if node is real or not
            self.rank = 0  # size of subtree that self is the root of (including self himself).

        else:  # Constructing leaf node.
            self.value = value
            self.left = AVLNode()
            self.right = AVLNode()
            self.left.parent = self
            self.right.parent = self
            self.parent = None
            self.height = 0  # height of node. BF uses this field. height of a leaf is 0.
            self.isVirtual = False  # field to see if node is real or not
            self.rank = 1  # size of subtree that self is the root of (including self himself).

    """
        returns the left child
        
        @rtype: AVLNode
        @returns: the left child of self, None if there is no left child
        
        Time Complexity - O(1)
    """

    def getLeft(self):
        return self.left  # if the node is virtual, then left is None

    """
        returns the right child
    
        @rtype: AVLNode
        @returns: the right child of self, None if there is no right child
        
        Time Complexity - O(1)
    """

    def getRight(self):
        return self.right  # if the node is virtual, then right is None

    """
        returns the parent 
    
        @rtype: AVLNode
        @returns: the parent of self, None if there is no parent
        
        Time Complexity - O(1)
    """

    def getParent(self):
        # if there is no parent, then parent is set to None.
        return self.parent

    """
        return the value
    
        @rtype: str
        @returns: the value of self, None if the node is virtual
        
        Time Complexity - O(1)
    """

    def getValue(self):
        # if the node is virtual then the value is set to None.
        return self.value

    """
        returns the height
    
        @rtype: int
        @returns: the height of self, -1 if the node is virtual
        
        Time Complexity - O(1)
    """

    def getHeight(self):
        # if the node is virtual then the height is set to -1.
        return self.height

    """
        returns the balance factor
    
        @rtype: int
        @returns: the balance factor of self, 3 if the node is virtual
        
        Time Complexity - O(1)
    """

    def getBF(self):
        if self.isVirtual:  # node is virtual
            return 3
        return self.left.height - self.right.height

    """
        returns the rank
    
        @rtype: int
        @returns: the rank of self, 0 if the node is virtual
        
        Time Complexity - O(1)
    """

    def getRank(self):
        # if the node is virtual then the height is set to 0.
        return self.rank

    """
        sets left child
    
        @type node: AVLNode
        @param node: a node
        
        Time Complexity - O(1)
    """

    def setLeft(self, node):
        self.left = node

    """
        sets right child
    
        @type node: AVLNode
        @param node: a node
        
        Time Complexity - O(1)
    """

    def setRight(self, node):
        self.right = node

    """
        sets parent
    
        @type node: AVLNode
        @param node: a node
        
        Time Complexity - O(1)
    """

    def setParent(self, node):
        self.parent = node

    """
        sets value
    
        @type value: str
        @param value: data
        
        Time Complexity - O(1)
    """

    def setValue(self, value):
        self.value = value

    """
        sets the balance factor of the node
    
        @type h: int
        @param h: the height
        
        Time Complexity - O(1)
    """

    def setHeight(self, h):
        self.height = h

    """
        sets the rank of the node
    
        @type r: int
        @param r: the rank
        
        Time Complexity - O(1)
    """
    def setRank(self, r):
        self.rank = r

    """
        returns whether self is not a virtual node 
    
        @rtype: bool
        @returns: False if self is a virtual node, True otherwise.
        
        Time Complexity - O(1)
    """

    def isRealNode(self):
        if self.isVirtual:
            return False
        return True


"""
    A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
        Constructor, you are allowed to add more fields.

        Time Complexity - O(1)
    """

    def __init__(self, root=None):
        self.root = root

        if root is None:
            self.size = 0
        else:
            self.size = root.getRank()

    """
        returns whether the list is empty
    
        @rtype: bool
        @returns: True if the list is empty, False otherwise
        
        Time Complexity - O(1)
    """

    def empty(self):
        if self.size == 0:
            return True
        return False

    """
        retrieves the *node* of the i'th item in the list

        @type i: int
        @pre: 0 <= i < self.length()
        @param i: index in the list
        @rtype: AVLNode
        @returns: the the *node* of the i'th item in the list
        
        Time Complexity - O(log(n))
    """

    def find_node(self, i):
        # implementing tree-select(i+1), but returns a node instead of value
        def find_node_rec(node, k):
            r = node.getLeft().getRank() + 1

            if k == r:
                return node

            elif k < r:
                return find_node_rec(node.getLeft(), k)

            else:
                return find_node_rec(node.getRight(), k - r)

        if i >= self.size or i < 0:
            return None
        return find_node_rec(self.root, i + 1)

    """
        retrieves the value of the i'th item in the list
    
        @type i: int
        @pre: 0 <= i < self.length()
        @param i: index in the list
        @rtype: str
        @returns: the the value of the i'th item in the list
        
        Time Complexity - O(log(n))
    """

    def retrieve(self, i):
        if i >= self.size or i < 0:
            return None
        node = self.find_node(i)
        return node.getValue()

    """
        inserts val at position i in the list
    
        @type i: int
        @pre: 0 <= i <= self.length()
        @param i: The intended index in the list to which we insert val
        @type val: str
        @param val: the value we inserts
        @rtype: list
        @returns: the number of rebalancing operation due to AVL rebalancing
        
        Time Complexity - O(log(n))
    """

    def insert(self, i, val):
        node = AVLNode(val)

        if self.empty():
            self.root = node
            self.size += 1
            return 0

        if i == 0:  # need to insert in the beginning of the list
            parent = self.find_node(i)
            parent.setLeft(node)
            node.setParent(parent)

        else:
            parent = self.find_node(i-1)

            if not parent.getRight().isRealNode():  # parent doesn't have right child
                parent.setRight(node)
                node.setParent(parent)

            else:  # needs to find the successor
                parent = parent.getRight()
                while parent.getLeft().isRealNode():  # going left until we reached node that don't have left child
                    parent = parent.getLeft()
                parent.setLeft(node)
                node.setParent(parent)

        self.size += 1

        self.fixRanks(parent, "insert")

        return self.rebalanceTree(node.getParent(), 0, "insert")

    """
        fix AVL tree after inserting node to it and returns the number of rebalancing operations 
    
        @type node: AVLNode
        @param node: the node to start rebalance the tree from
        @type count: int
        @param count: the number of rebalancing operations
        @type in_or_out: str
        @param in_or_out: rebalance after insertion ot deletion
        @rtype: int
        @returns: the number of rebalancing operations
        
        Time Complexity - O(log(n))
    """

    def rebalanceTree(self, node, count, in_or_out):

        while node is not None:
            # calc height
            new_height = max(node.getRight().getHeight(), node.getLeft().getHeight()) + 1
            if node.getHeight() != new_height:
                node.setHeight(new_height)
                height_has_changed = True
            else:
                height_has_changed = False

            # calc BF
            new_bf = node.getLeft().getHeight() - node.getRight().getHeight()

            # observation 1
            if -1 <= new_bf <= 1 and not height_has_changed:
                if in_or_out == "delete":
                    node = node.getParent()
                else:
                    break

            # observation 2
            elif -1 <= new_bf <= 1 and height_has_changed:
                node = node.getParent()

            else:  # observation 3
                # perform rotations
                if in_or_out == "insert":
                    if new_bf == 2:
                        if node.getLeft().getBF() == -1:
                            self.rotateLeft(node.getLeft())
                            self.rotateRight(node)
                            count += 1
                        elif node.getLeft().getBF() == 1:
                            self.rotateRight(node)

                    elif new_bf == -2:
                        if node.getRight().getBF() == -1:
                            self.rotateLeft(node)
                        elif node.getRight().getBF() == 1:
                            self.rotateRight(node.getRight())
                            self.rotateLeft(node)
                            count += 1

                    count += 1
                    break  # one rotate fixes it all in insertion

                if in_or_out == "delete":
                    if new_bf == 2:
                        if node.getLeft().getBF() == -1:
                            self.rotateLeft(node.getLeft())
                            self.rotateRight(node)
                            count += 1
                        elif node.getLeft().getBF() == 1 or node.getLeft().getBF() == 0:
                            self.rotateRight(node)

                    elif new_bf == -2:
                        if node.getRight().getBF() == -1 or node.getRight().getBF() == 0:
                            self.rotateLeft(node)
                        elif node.getRight().getBF() == 1:
                            self.rotateRight(node.getRight())
                            self.rotateLeft(node)
                            count += 1

                    count += 1
                    node = node.getParent()

        return count

    """
        fix the ranks of tree after inserting or deleting node to it

        @type node: AVLNode
        @param node: the node to start fixing the tree from
        @type in_or_out: str
        @param in_or_out: fix ranks after insertion ot deletion
        
        Time Complexity - O(log(n))
    """

    def fixRanks(self, node, in_or_out):
        if in_or_out == "insert":
            add = 1
        if in_or_out == "delete":
            add = -1

        # changing the rank
        while node is not None:
            if in_or_out == "concat":
                new_rank = node.getLeft().getRank() + node.getRight().getRank() + 1
            else:
                new_rank = node.getRank() + add
            node.setRank(new_rank)
            node = node.getParent()

        return None

    """
        deletes the i'th item in the list
    
        @type i: int
        @pre: 0 <= i < self.length()
        @param i: The intended index in the list to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
        
        Time Complexity - O(log(n))
    """

    def delete(self, i):

        if i >= self.size or i < 0:
            return -1

        """
        if i==11:
            self.printTreeByLevels(self.getTreeHeight())
            print(self.listToArray())
        """

        # this is the node we would like to delete
        node = self.find_node(i)

        # first - what if node is the root?
        if self.root == node:
            if self.size == 1:
                self.setRoot(AVLNode())  # tree is now empty
                self.size = 0
                return 0

            # observation - if root has only one child - must be a tree with size = 2
            # we simply make the child our new root
            if self.size == 2:
                if node.getRight().isRealNode():  # left son is a leaf
                    self.root = node.getRight()
                    self.root.setParent(None)
                    node.setRight(AVLNode())
                    node.getRight().setParent(node)
                    self.root.setRank(1)
                elif node.getLeft().isRealNode():  # right son is a leaf
                    self.root = node.getLeft()
                    self.root.setParent(None)
                    node.setLeft(AVLNode())
                    node.getLeft().setParent(node)
                    self.root.setRank(1)
                self.size -= 1
                return 0

            else:  # size > 2
                # looking for the successor!
                succ = node.getRight()
                while succ.getLeft().getValue() is not None:
                    succ = succ.getLeft()

                # connecting between succ's parent and succ's child
                succ_parent = succ.getParent()
                succ_child = succ.getRight()  # might be virtual
                succ_child.setParent(succ_parent)

                if succ_parent.getLeft() == succ:
                    succ_parent.setLeft(succ_child)
                    succ_is_node_son = False
                else:  # succ is the right son of node to be deleted
                    succ_parent.setRight(succ_child)
                    succ_is_node_son = True

                succ.setLeft(node.getLeft())
                node.getLeft().setParent(succ)
                succ.setRank(node.getRank())
                succ.setRight(node.getRight())
                node.getRight().setParent(succ)

                if not succ_is_node_son:
                    fix_from_here = succ_parent

                else:
                    fix_from_here = succ

                succ.setParent(None)  # became the new root
                self.setRoot(succ)

        # if node is not the root
        else:
            parent = node.getParent()

            # checking whether node is right or left child to his parent
            if parent.getLeft() == node:
                left_child = True
                right_child = False
            if parent.getRight() == node:
                right_child = True
                left_child = False

            if node.getRank() <= 2:

                # case 1 - node is a leaf
                if node.getRank() == 1:
                    if left_child:
                        parent.setLeft(AVLNode())
                        parent.getLeft().setParent(parent)
                    else:  # node is the right son
                        parent.setRight(AVLNode())
                        parent.getRight().setParent(parent)

                # case 2 - node has only one child
                elif not node.getLeft().isRealNode():  # it's a right son
                    if left_child:
                        parent.setLeft(node.getRight())
                    elif right_child:
                        parent.setRight(node.getRight())
                    node.getRight().setParent(parent)

                elif not node.getRight().isRealNode():  # it's a left son
                    if left_child:
                        parent.setLeft(node.getLeft())
                    elif right_child:
                        parent.setRight(node.getLeft())
                    node.getLeft().setParent(parent)

                fix_from_here = parent

            # case 3 - node has 2 children
            if node.getLeft().isRealNode() and node.getRight().isRealNode():
                # looking for the successor!
                succ = node.getRight()
                while succ.getLeft().getValue() is not None:
                    succ = succ.getLeft()

                # connecting between succ's parent and succ's child
                succ_parent = succ.getParent()
                succ_child = succ.getRight()  # might be virtual
                succ_child.setParent(succ_parent)

                if succ_parent.getLeft() == succ:
                    succ_parent.setLeft(succ_child)
                    succ_is_node_son = False
                else:  # succ is the right son of node to be deleted
                    succ_parent.setRight(succ_child)
                    succ_is_node_son = True

                succ.setLeft(node.getLeft())
                node.getLeft().setParent(succ)
                succ.setRank(node.getRank())
                succ.setRight(node.getRight())
                node.getRight().setParent(succ)

                if not succ_is_node_son:
                    fix_from_here = succ_parent

                else:
                    fix_from_here = succ

                # connecting between succ and deleted node's parent and children
                succ.setParent(parent)
                if left_child:
                    parent.setLeft(succ)
                elif right_child:
                    parent.setRight(succ)

        self.size -= 1

        self.fixRanks(fix_from_here, "delete")
        return self.rebalanceTree(fix_from_here, 0, "delete")

    """
        returns the value of the first item in the list
    
        @rtype: str
        @returns: the value of the first item, None if the list is empty
        
        Time Complexity - O(log(n))
    """

    def first(self):
        if self.empty():
            return None
        if self.size == 1:
            return self.getRoot().getValue()

        # going all the way to left
        node = self.getRoot()
        while node.isRealNode() and node.getLeft().isRealNode():
            node = node.getLeft()

        return node.getValue()

    """
        returns the value of the last item in the list
    
        @rtype: str
        @returns: the value of the last item, None if the list is empty
        
        Time Complexity - O(log(n))
    """

    def last(self):
        if self.empty():  # list is empty
            return None

        node = self.root

        while node.getRight().isRealNode():  # stops when reached node that doesn't have right child
            node = node.getRight()

        return node.getValue()

    """
        returns an array representing list 
    
        @rtype: list
        @returns: a list of strings representing the data structure
        
        Time Complexity - O(n)
    """

    def listToArray(self):
        listArr = []

        if self.empty():  # list is empty
            return listArr

        self.inorderToArray(self.getRoot(), listArr)

        return listArr

    """
        adds to an array the representation of a list going in order through the list

        @type node: AVLNode
        @param node: the root of a tree that needs to be represented
        @type arr: list
        @param arr: the list that represent the tree
        
        Time Complexity - O(n)
    """

    def inorderToArray(self, node, arr):
        if node is None or not node.isRealNode():
            return

        if not node.getLeft().isRealNode():  # there isn't any more left children
            arr.append(node.getValue())
            self.inorderToArray(node.getRight(), arr)

        else:
            self.inorderToArray(node.getLeft(), arr)
            arr.append(node.getValue())
            self.inorderToArray(node.getRight(), arr)

        return

    """
        returns the size of the list 
    
        @rtype: int
        @returns: the size of the list
        
        Time Complexity - O(1)
    """

    def length(self):
        return self.size

    """
        sort the info values of the list
    
        @rtype: list
        @returns: an AVLTreeList where the values are sorted by the info of the original list.
        
        Time Complexity - O(nlog n)
    """

    def sort(self):

        MIN_MERGE = 32

        # Returns the minimum length of a run from 23 to 64
        # so that the len(array)/MinRun is less than or equal to a power of 2.
        def calcMinRun(n):
            r = 0
            while n >= MIN_MERGE:
                r |= n & 1
                n >>= 1
            return n + r

        # This function sorts array from left index to right index which is of size at most RUN
        def insertionSort(arr, left, right):
            for i in range(left + 1, right + 1):
                j = i
                while j > left and arr[j] < arr[j - 1]:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]
                    j -= 1

        # Merge function merges the sorted runs
        # l = left, m = middle, r = right
        def merge(arr, l, m, r):

            # original array is broken into two parts
            # left and right array
            len1, len2 = m - l + 1, r - m
            left, right = [], []
            for i in range(0, len1):
                left.append(arr[l + i])
            for i in range(0, len2):
                right.append(arr[m + 1 + i])

            i, j, k = 0, 0, l

            # after comparing, we merge those two array in larger sub-array
            while i < len1 and j < len2:
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1

                else:
                    arr[k] = right[j]
                    j += 1

                k += 1

            # Copy remaining elements of left, if any
            while i < len1:
                arr[k] = left[i]
                k += 1
                i += 1

            # Copy remaining element of right, if any
            while j < len2:
                arr[k] = right[j]
                k += 1
                j += 1

        # Iterative Timsort function to sort the array[0...n-1] (similar to merge sort)
        def timSort(arr):
            n = len(arr)
            minRun = calcMinRun(n)

            # Sort individual sub-arrays of size RUN
            for start in range(0, n, minRun):
                end = min(start + minRun - 1, n - 1)
                insertionSort(arr, start, end)

            # Start merging from size RUN (or 32). It will merge
            # to form size 64, then 128, 256 and so on ....
            size = minRun
            while size < n:

                # Pick starting point of left sub array.
                # We are going to merge arr[left..left+size-1] and arr[left+size, left+2*size-1].
                # After every merge, we increase left by 2*size
                for left in range(0, n, 2 * size):

                    # Find ending point of left sub array
                    # mid+1 is starting point of right sub array
                    mid = min(n - 1, left + size - 1)
                    right = min((left + 2 * size - 1), (n - 1))

                    # Merge sub array arr[left.....mid] & arr[mid+1....right]
                    if mid < right:
                        merge(arr, left, mid, right)

                size = 2 * size

        listArr = self.listToArray()

        timSort(listArr)

        ret_tree = self.copyTree()
        index = [0]
        # second step - inserting the values in the new order to the copy tree
        ret_tree.inorder_insert_rec(ret_tree.getRoot(), listArr, index)

        return ret_tree

    """
    permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    
    Time Complexity - O(n)
    """

    def permutation(self):
        # first step: creating an array of the values in a random order
        def perm_rec(node, l):
            # choosing the order of the values in the array:
            # 0 - go left
            # 1 - go right
            # 2 - append
            options = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
            num = random.randrange(0, 6)
            for i in range(3):
                if options[num][i] == 0 and node.getLeft().isRealNode():
                    perm_rec(node.getLeft(), l)
                elif options[num][i] == 1 and node.getRight().isRealNode():
                    perm_rec(node.getRight(), l)
                elif options[num][i] == 2:
                    l.append(node.getValue())

            return l

        if self.size == 0:  # tree is empty
            return AVLTreeList()

        res = []
        perm_rec(self.getRoot(), res)

        ret_tree = self.copyTree()
        index = [0]
        # second step - inserting the values in the new order to the copy tree
        ret_tree.inorder_insert_rec(ret_tree.getRoot(), res, index)

        return ret_tree

    """
         inserting the values into an empty existing tree

            @type lst: AVLTreeList, 
            @param lst: node, list, int index
            @returns: None

            Time Complexity - O(n)       
     """

    def inorder_insert_rec(self, node, l, index):
        if node.getLeft().isRealNode():
            self.inorder_insert_rec(node.getLeft(), l, index)
        node.setValue(l[index[0]])
        index[0] += 1
        if node.getRight().isRealNode():
            self.inorder_insert_rec(node.getRight(), l, index)

        return None

    """
        copying the form of our tree, with default value of -1

        @type lst: AVLTreeList
        @rtype: AVLTreeList
        @returns: a new AVLTreeList which is a copy of the original, but all values are -1.
        
        Time Complexity - O(n)
    """

    def copyTree(self):

        def copyTreeRec(org_node, copy_node):
            if org_node.getLeft().isRealNode():
                new_left_node = copyNode(org_node.getLeft(), copy_node)
                copy_node.setLeft(new_left_node)
                copyTreeRec(org_node.getLeft(), new_left_node)
            if org_node.getRight().isRealNode():
                new_right_node = copyNode(org_node.getRight(), copy_node.getParent())
                copy_node.setRight(new_right_node)
                copyTreeRec(org_node.getRight(), new_right_node)

            return None

        def copyNode(node, copy_parent):
            copy = AVLNode(-1)
            copy.setParent(copy_parent)
            copy.setHeight(node.getHeight())
            copy.setRank(node.getRank())
            return copy

        new_tree = AVLTreeList()
        if self.size == 0:  # an empty tree
            return new_tree

        new_tree.setRoot(copyNode(self.root, None))
        copyTreeRec(self.root, new_tree.root)
        new_tree.size = self.size
        return new_tree

    """
        concatenates lst to self
    
        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined
        
        Time Complexity - O(log(n))
    """

    def concat(self, lst):
        if self.size == 0 or lst.size == 0:
            diff = lst.getTreeHeight() - self.getTreeHeight()
            if self.size == 0 and lst.size != 0:
                self.setRoot(lst.root)
            return abs(diff)

        our_height = self.root.getHeight()
        lst_height = lst.root.getHeight()
        diff = our_height - lst_height

        new_size = self.size + lst.size + 1

        x = AVLNode("joiner")
        will_be_deleted_index = self.size

        # step 1 - check who is higher
        if diff >= 0:    # self is higher
            pointer = self.root

            for i in range(abs(diff)):
                pointer = pointer.getRight()

            # connecting the joiner between the two trees
            parent = pointer.getParent()
            if parent is not None:
                parent.setRight(x)
            x.setParent(parent)
            x.setLeft(pointer)
            pointer.setParent(x)
            x.setRight(lst.root)
            lst.root.setParent(x)

            x.setHeight(pointer.getHeight() + 1)

            # check whether we need to rebalance
            x.setRank(x.getRight().getRank() + x.getLeft().getRank() + 1)
            if parent is not None and abs(parent.getBF()) >= 2:
                self.rebalanceTree(parent, 0, "delete")

        else:     # lst is higher
            pointer = lst.root

            for i in range(abs(diff)):
                pointer = pointer.getLeft()

            # connecting the joiner between the two trees
            parent = pointer.getParent()
            if parent is not None:
                parent.setLeft(x)
            x.setParent(parent)
            x.setRight(pointer)
            pointer.setParent(x)
            x.setLeft(self.root)
            self.root.setParent(x)

            x.setHeight(pointer.getHeight() + 1)

            # check whether we need to rebalance
            x.setRank(x.getRight().getRank() + x.getLeft().getRank() + 1)
            self.root = lst.root
            if parent is not None and abs(parent.getBF()) >= 2:
                self.rebalanceTree(parent, 0, "delete")

        # now we can delete the joiner x
        if diff == 0:
            self.root = x
        self.size = new_size
        self.fixRanks(x, "concat")
        self.delete(will_be_deleted_index)

        return abs(diff)

    """
        searches for a *value* in the list
    
        @type val: str
        @param val: a value to be searched
        @rtype: int
        @returns: the first index that contains val, -1 if not found.
        
        Time Complexity - O(n)
    """

    def search(self, val):
        if self.empty():  # the list is empty
            return -1

        listArr = self.listToArray()

        try:
            index = listArr.index(val)
        except ValueError:  # val isn't in the list
            index = -1

        return index

    """
        returns the root of the tree representing the list
    
        @rtype: AVLNode
        @returns: the root, None if the list is empty
        
        Time Complexity - O(1)
    """

    def getRoot(self):
        # if the list is empty then root is set to None.
        return self.root

    """
        returns the height of the tree

        @rtype: int
        @returns: height of the tree
        
        Time Complexity - O(1)
    """
    def getTreeHeight(self):
        if self.empty():  # list is empty
            return -1
        return self.root.getHeight()

    """
        sets the root of the tree representing the list

        @type root: AVLNode
        @param root: a node
        
        Time Complexity - O(1)
    """

    def setRoot(self, root):
        if not root.isRealNode():  # root is virtual node
            self.root = None
            self.size = 0
            return None

        self.root = root
        self.size = root.getRank()
        return None

    """
        rotates to the right on self
        
        rotation to right changes the height and rank of only self and his left child.
        
        @type node: AVLNode
        @param node: node to do the rotation on
        
        Time Complexity - O(1)
    """

    def rotateRight(self, node):
        x = node
        y = x.getLeft()

        # moving the right subtree of y to be the left subtree of x
        x.setLeft(y.getRight())
        y.getRight().setParent(x)

        # changing y right child to be x
        y.setRight(x)

        # changing the parent of y to be x.parent
        y.setParent(x.getParent())

        # changing y.parent child from x to y
        if y.getParent() is not None:  # x wasn't the root
            setNewRoot = False
            if y.getParent().getRight() == x:  # x was the right child of his parent
                y.getParent().setRight(y)
            elif y.getParent().getLeft() == x:  # x was the left child of his parent
                y.getParent().setLeft(y)
        else:  # y was the root, need to update the root
            setNewRoot = True
            newRoot = y

        # changing y to be x's parent
        x.setParent(y)

        # setting the new heights and ranks

        # the height is the max height between his children + 1
        heightX = max(x.getRight().getHeight(), x.getLeft().getHeight()) + 1
        heightY = max(heightX, y.getLeft().getHeight()) + 1  # x is y's right child
        # the rank is the sum of his children ranks + 1
        rankX = x.getRight().getRank() + x.getLeft().getRank() + 1
        rankY = rankX + y.getLeft().getRank() + 1  # x is y's right child

        y.setHeight(heightY)
        y.setRank(rankY)
        x.setHeight(heightX)
        x.setRank(rankX)

        if setNewRoot:
            self.setRoot(newRoot)

        return None

    """
        rotates to the left on self

        rotation to left changes the height and rank of only self and his right child.
        
        @type node: AVLNode
        @param node: node to do the rotation on
        
        Time Complexity - O(1)
    """

    def rotateLeft(self, node):
        y = node
        x = y.getRight()

        # moving the left subtree of x to be the right subtree of y
        y.setRight(x.getLeft())
        x.getLeft().setParent(y)

        # changing x left child to be y
        x.setLeft(y)

        # changing the parent of x to be y.parent
        x.setParent(y.getParent())

        # changing x.parent child from y to x
        if x.getParent() is not None:  # y wasn't the root
            setNewRoot = False
            if x.getParent().getRight() == y:  # y was the right child of his parent
                x.getParent().setRight(x)
            elif x.getParent().getLeft() == y:  # y was the left child of his parent
                x.getParent().setLeft(x)
        else:  # y was the root, need to update the root
            setNewRoot = True
            newRoot = x

        # changing x to be y's parent
        y.setParent(x)

        # setting the new heights and ranks

        # the height is the max height between his children + 1
        heightY = max(y.getRight().getHeight(), y.getLeft().getHeight()) + 1
        heightX = max(y.getRight().getHeight(), heightY) + 1  # y is x's left child
        # the rank is the sum of his children ranks + 1
        rankY = y.getRight().getRank() + y.getLeft().getRank() + 1
        rankX = x.getRight().getRank() + rankY + 1  # y is x's left child

        y.setHeight(heightY)
        y.setRank(rankY)
        x.setHeight(heightX)
        x.setRank(rankX)

        if setNewRoot:
            self.setRoot(newRoot)

        return


