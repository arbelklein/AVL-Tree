# AVL-Tree

## Description
This repository contains a university course project that focuses on implementing an abstract data type (ADT) list using an AVL tree.

## Usage
To use the AVL tree-based list implementation:
1. Import the ` AVLTreeList` class from the provided `AVLTreeList.py` file.
2. Create an instance of `AVLTreeList`:
    ```python
    from AVLTreeList import AVLTreeList
    avl_list = AVLTreeList()
    ```
3. Use the available methods to interact with the list. For example:
    ```python
    avl_list.insert(0, "first_element")
    avl_list.insert(1, "second_element")
    print(avl_list.retrieve(0))  # Output: "first_element"
    print(avl_list.length())     # Output: 2
    ```

## Output
The implementation does not include direct user interaction or terminal output, as it is designed to be used programmatically. However, the methods return values that reflect the state of the AVL tree-based list. Examples include:
- `empty()`:
  ```python
  print(avl_list.empty())  # Output: True (if the list is empty)
  ```
- `insert()`:
  ```python
  balance_ops = avl_list.insert(0, "element")
  print(balance_ops)  # Output: Number of balance operations performed
  ```
- listToArray():
  ```python
  print(avl_list.listToArray())  # Output: ["element1", "element2", ...]
  ```

These outputs can be used to verify the correctness and behavior of the implemented methods during testing and experimentation.

