import re
import pandas as pd

pattern = re.compile(r'@[\w]+')
pattern_link = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")


def handle_data(row: pd.Series) -> str:
    text: str = row['tweet_text']
    query: str = row['query_used']
    text = text.replace(query, '')
    text = re.sub(pattern, '', text)
    text = re.sub(pattern_link, '', text)
    return text.strip()


test_data = pd.read_csv('twitter/data/Test.csv', sep=';')
train_data = pd.read_csv('twitter/data/Train100.csv', sep=';')

test_data['task'] = 'test'
train_data['task'] = 'train'

print(f"test: {len(test_data)} train: {len(train_data)}")
full_data = pd.concat([test_data, train_data], ignore_index=True)
full_data['text'] = full_data.apply(handle_data, axis=1)

full_data[['text']].to_csv('data/corpus/twt_pt.txt', sep='\t', header=False, index=False)
full_data[['task', 'sentiment']].to_csv('data/twt_pt.txt', sep='\t', header=False)
print('files saved')