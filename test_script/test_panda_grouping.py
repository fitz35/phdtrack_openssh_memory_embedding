import pandas as pd

def transform_dict(input_dict):
    """
    Transforms a dictionary with tuple keys into a nested dictionary.
    
    Args:
    - input_dict (dict): A dictionary with tuple keys (int, str) and int values.
    
    Returns:
    - dict: A nested dictionary where the first element of the tuple becomes the outer key
            and the second element becomes the inner key.
    """
    
    # Initialize the output dictionary
    output_dict = {}
    
    # Iterate over each key, value pair in the input dictionary
    for (num, letter), value in input_dict.items():
        
        # If the number (num) is not already a key in the output dictionary, add it
        if num not in output_dict:
            output_dict[num] = {}
        
        # Add the letter as a key inside the inner dictionary and set its value
        output_dict[num][letter] = value
        
    return output_dict

clusters = pd.Series([1, 2, 1, 2, 3, 3, 1], name='Clusters')
labels = pd.Series(['A', 'B', 'A', 'A', 'B', 'C', 'C'], name='Labels')
df = pd.DataFrame({'Cluster': clusters, 'Label': labels})
counts = df.groupby(['Cluster', 'Label']).size()
counts_dict = transform_dict(counts.to_dict())

print(counts_dict)