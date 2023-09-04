# %%

import pandas as pd
with open("index.md", "r") as f:
    markdown = f.readlines()

# %%

markdown = pd.DataFrame(markdown)
markdown.columns = ['text']

# %%
markdown['content'] = ~markdown['text'].str.contains(r'^\s*$', regex=True)

# %%
markdown

# %%
markdown = markdown.reset_index()

# %%
content_df = markdown[markdown['content']].reset_index(drop=True)

# %%
content_df

# %%
content = content_df['text'].values

# %%
for i in range(len(content)):
    if "{{< /details >}}" in content[i] and "![png]" in content[i-1]:
        print(content[i-1], content[i])
        print(content_df['index'][i-1], content_df['index'][i])
        
        markdown['text'][content_df['index'][i-1]] = content[i]
        markdown['text'][content_df['index'][i]] = content[i-1]

# %%
new_text = "".join(markdown['text'])

with open("index.md", "w") as f:
    markdown = f.write(new_text)


