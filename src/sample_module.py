# sample_module.py

import numpy as np
import pandas as pd

class SampleClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

def sample_function(x, y):
    return x + y

def create_dataframe():
    data = {
        'Column1': [1, 2, 3, 4],
        'Column2': [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Sample usage
    obj = SampleClass("World")
    print(obj.greet())
    
    print("Sum:", sample_function(3, 4))
    
    arr = np.array([1, 2, 3, 4])
    print("Numpy Array:", arr)
    
    df = create_dataframe()
    print("DataFrame:\n", df)
