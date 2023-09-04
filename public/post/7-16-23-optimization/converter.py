# Import necessary libraries
import os
import pandas as pd

# Convert a Jupyter Notebook to Markdown using os.system
os.system("jupyter nbconvert --to markdown index.ipynb")

# Read the generated Markdown file into a list of lines
with open("index.md", "r") as f:
    markdown = f.readlines()

# Create a Pandas DataFrame from the list of Markdown lines
markdown = pd.DataFrame(markdown)
markdown.columns = ['text']

# Create a boolean column 'content' to mark non-empty lines
markdown['content'] = (
    ~markdown['text'].str.contains(r'^\s*$', regex=True)
)

# Reset the index of the DataFrame
markdown = markdown.reset_index()

# Create a new DataFrame 'content_df' containing only non-empty lines
content_df = markdown[markdown['content']].reset_index(drop=True)
content = content_df['text'].values

# Iterate through the content to find specific patterns and swap lines
for i in range(len(content)):
    if "{{< /details >}}" in content[i] and "![png]" in content[i-1]:
        print(content[i-1], content[i])
        print(content_df['index'][i-1], content_df['index'][i])
        
        # Swap the lines in the original DataFrame
        markdown['text'][content_df['index'][i-1]] = content[i]
        markdown['text'][content_df['index'][i]] = content[i-1]

# Join the modified text back into a single string
new_text = "".join(markdown['text'])

# Write the modified content back to the original Markdown file
with open("index.md", "w") as f:
    markdown = f.write(new_text)
