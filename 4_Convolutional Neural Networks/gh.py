class BinarySearchTreeSort:
  def __init__(self, value):
    self.left = None
    self.value = value
    self.right = None
  def checkValue(self, value):
    if value < self.value:
       if self.left is None:
           self.left = BinarySearchTreeSort(value)
       else:
           self.left.checkValue(value)
    else:
       if self.right is None:
          self.right = BinarySearchTreeSort(value)
       else:
          self.right.checkValue(value)
  @classmethod
  def display(cls, _node):
     return list(filter(None, [i for b in [cls.display(_node.left) if isinstance(_node.left, BinarySearchTreeSort) else [getattr(_node.left, 'value', None)], [_node.value], cls.display(_node.right) if isinstance(_node.right, BinarySearchTreeSort) else [getattr(_node.right, 'value', None)]] for i in b]))


bst = BinarySearchTreeSort(4)

arr =[21, 31, 52,  9, 57, 35, 62, 94, 50,  1, 50, 40, 39, 23, 30, 40, 58,
       39, 14, 75]

for i in arr:
  bst.checkValue(i)

print(BinarySearchTreeSort.display(bst))