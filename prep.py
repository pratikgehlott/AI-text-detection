import pandas as pd
from uuid import uuid4
import pandas as pd
import json
# Plot histogram and check the cutoff
TOKEN_LENGTH_CAP = 200
# CHANGE FILE AS REQUIRED
file_path = 'data_AskCulinary.csv'


with open('results.json','w') as f:
    json.dump({}, f)
data = pd.read_csv(file_path)
data['id'] = [str(uuid4()) for _ in range(len(data.index))]
data['chatgpt-3.5-turbo'] = '[EMPTY]'
data = data[data['selftext'] != '[deleted]']
data['text'] = data['selftext'].map(lambda x: ' '.join(x.split()[:TOKEN_LENGTH_CAP]))
data['text'] = data['text'].map(lambda x: x.replace('\n', ' '))
data.to_csv(file_path, index=False)