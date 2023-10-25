from typing import Tuple
from sklearn.utils import shuffle
import pandas as pd

# Assuming df is your features DataFrame and labels is your series
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
})
labels = pd.Series([0, 1, 2, 0, 0, 0, 0, 0, 4, 1])

import pandas as pd
from sklearn.utils import shuffle

def balance_classes(df: pd.DataFrame, labels: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balances the classes in the given DataFrame and Series by undersampling the majority class.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the features.
    - labels: pd.Series
        The input Series containing the labels.
    - random_state: int, optional (default=42)
        Controls the randomness of the sampling and shuffling.
        
    Returns:
    - tuple[pd.DataFrame, pd.Series]
        The balanced and shuffled DataFrame and Series.
    """
    # Separate the majority and minority classes
    df_minority = df[labels != 0]
    labels_minority = labels[labels != 0]
    
    df_majority = df[labels == 0]
    labels_majority = labels[labels == 0]
    
    # Check if minority classes are actually in minority
    if len(df_minority) >= len(df_majority):
        return df, labels
    
    # Sample from the majority class to balance the classes
    df_majority_sampled = df_majority.sample(n=len(df_minority), random_state=random_state)
    labels_majority_sampled = labels_majority.loc[df_majority_sampled.index]
    
    # Concatenate the majority and minority samples
    df_sampled = pd.concat([df_minority, df_majority_sampled])
    labels_sampled = pd.concat([labels_minority, labels_majority_sampled])
    
    # Shuffle the data
    shufled_data : Tuple[pd.DataFrame, pd.Series] = shuffle(df_sampled, labels_sampled, random_state=random_state) # type: ignore
    df_sampled, labels_sampled = shufled_data
    
    return df_sampled, labels_sampled

df_sampled, labels_sampled = balance_classes(df, labels)


print(df_sampled)
print(labels_sampled)