import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from IPython.core.display import display, HTML
from sklearn.datasets import load_iris

# Load iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display first 10 rows and info
print(df.head(10))
print(df.info())

# Generate profiling report
prof = ProfileReport(df, title="Iris Dataset Profiling Report")
prof.to_file(output_file='EDA.html')

# Display report in Jupyter notebook (if applicable)
display(HTML(prof.to_html()))
