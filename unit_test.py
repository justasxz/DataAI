import pytest

def function_to_test(x):
    return x * 2

# let's use pytest to test this function
def test_function_to_test():
    assert function_to_test(2) == 4
    assert function_to_test(-1) == -2
    assert function_to_test(0) == 0