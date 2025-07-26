import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
data = pd.read_csv('netflix1.csv')

data.drop_duplicates(inplace=True)
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month
data['genres'] = data['listed_in'].apply(lambda x: x.split(', '))
data['duration_type'] = data['duration'].apply(lambda x: x.split(' ')[-1])
data['duration_int'] = data['duration'].apply(lambda x: ''.join(filter(str.isdigit, x))).astype(int)

plt.figure(figsize=(6,4))
sns.countplot(data=data, x='type', palette='Set2')
plt.title('Distribution of Movies and TV Shows')
plt.show()

from itertools import chain
genre_series = pd.Series(list(chain.from_iterable(data['genres']))).value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=genre_series.values, y=genre_series.index, palette='Set3')
plt.title('Top 10 Most Common Genres')
plt.xlabel('Count')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(data=data, x='year_added', order=sorted(data['year_added'].dropna().unique()), palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Content Added Over the Years')
plt.xlabel('Year Added')
plt.ylabel('Count')
plt.show()

top_directors = data['director'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(y=top_directors.index, x=top_directors.values, palette='Blues_d')
plt.title('Top 10 Directors with Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()

movie_titles = data[data['type'] == 'Movie']['title']
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Titles")
plt.show()

print("Data cleaned, analyzed, and visualized successfully!")
data.to_csv("cleaned_netflix_data.csv", index=False)
