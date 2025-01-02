import pandas as pd
import numpy as np

def generate_iris_rows(num_rows):
    # Define ranges for the Iris dataset based on general values
    sepal_length = np.random.uniform(4.3, 7.9, num_rows)  # Range for sepal length
    sepal_width = np.random.uniform(2.0, 4.4, num_rows)   # Range for sepal width
    petal_length = np.random.uniform(1.0, 6.9, num_rows)  # Range for petal length
    petal_width = np.random.uniform(0.1, 2.5, num_rows)   # Range for petal width
    target = np.random.choice([0, 1], num_rows)        # Target (classification)

    # Create a DataFrame
    iris_data = pd.DataFrame({
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width,
        'Target': target
    })

    # Round values to match the typical dataset format
    iris_data = iris_data.round(1)

    return iris_data

# Generate 10 rows as an example
iris_sample = generate_iris_rows(1000)

# Save to CSV if needed
iris_sample.to_csv("iris_sampleExtended.csv", index=False)
