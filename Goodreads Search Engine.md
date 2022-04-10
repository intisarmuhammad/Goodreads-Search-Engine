# Book Search Engine
This search engine will search a large book dataset scraped from the goodreads website by using their developer API (dataset from https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) and display all of its information. The information we can collect from this dataset is the book ID, title, author, average rating, language, number of pages, number of ratings, number of text reviews, publication date, and publisher.


```python
#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import re
import sklearn
```

# Importing Data


```python
#open csv file
book_data = pd.read_csv("/Users/intisarmuhammad/Downloads/books.csv")
```


```python
#view data of first 5 rows
book_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>eng</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>9/16/06</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>eng</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>9/1/04</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>eng</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>11/1/03</td>
      <td>Scholastic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>eng</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>5/1/04</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>eng</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>9/13/04</td>
      <td>Scholastic</td>
    </tr>
  </tbody>
</table>
</div>




```python
book_data.shape
```




    (11123, 10)



# Data Cleaning


```python
# create new title column for modified title names (remove non alphanumeric characters)
# this will make search queries easier for our search engine
book_data["mod_title"] = book_data["title"].str.replace("[^a-zA-Z0-9 ]", "", regex = True)
```


```python
book_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
      <th>mod_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>eng</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>9/16/06</td>
      <td>Scholastic Inc.</td>
      <td>Harry Potter and the HalfBlood Prince Harry Po...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>eng</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>9/1/04</td>
      <td>Scholastic Inc.</td>
      <td>Harry Potter and the Order of the Phoenix Harr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>eng</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>11/1/03</td>
      <td>Scholastic</td>
      <td>Harry Potter and the Chamber of Secrets Harry ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>eng</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>5/1/04</td>
      <td>Scholastic Inc.</td>
      <td>Harry Potter and the Prisoner of Azkaban Harry...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>eng</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>9/13/04</td>
      <td>Scholastic</td>
      <td>Harry Potter Boxed Set  Books 15 Harry Potter  15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11118</th>
      <td>45631</td>
      <td>Expelled from Eden: A William T. Vollmann Reader</td>
      <td>William T. Vollmann/Larry McCaffery/Michael He...</td>
      <td>4.06</td>
      <td>eng</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>12/21/04</td>
      <td>Da Capo Press</td>
      <td>Expelled from Eden A William T Vollmann Reader</td>
    </tr>
    <tr>
      <th>11119</th>
      <td>45633</td>
      <td>You Bright and Risen Angels</td>
      <td>William T. Vollmann</td>
      <td>4.08</td>
      <td>eng</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>12/1/88</td>
      <td>Penguin Books</td>
      <td>You Bright and Risen Angels</td>
    </tr>
    <tr>
      <th>11120</th>
      <td>45634</td>
      <td>The Ice-Shirt (Seven Dreams #1)</td>
      <td>William T. Vollmann</td>
      <td>3.96</td>
      <td>eng</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>8/1/93</td>
      <td>Penguin Books</td>
      <td>The IceShirt Seven Dreams 1</td>
    </tr>
    <tr>
      <th>11121</th>
      <td>45639</td>
      <td>Poor People</td>
      <td>William T. Vollmann</td>
      <td>3.72</td>
      <td>eng</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2/27/07</td>
      <td>Ecco</td>
      <td>Poor People</td>
    </tr>
    <tr>
      <th>11122</th>
      <td>45641</td>
      <td>Las aventuras de Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>3.91</td>
      <td>spa</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>5/28/06</td>
      <td>Edimat Libros</td>
      <td>Las aventuras de Tom Sawyer</td>
    </tr>
  </tbody>
</table>
<p>11123 rows × 11 columns</p>
</div>




```python
# transform modified titles to lower case
book_data["mod_title"] = book_data["mod_title"].str.lower()
```


```python
# remove multiple spaces in a row
book_data["mod_title"] = book_data["mod_title"].str.replace("\s+", " ", regex = True)
```


```python
#view the data
book_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
      <th>mod_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>eng</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>9/16/06</td>
      <td>Scholastic Inc.</td>
      <td>harry potter and the halfblood prince harry po...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>eng</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>9/1/04</td>
      <td>Scholastic Inc.</td>
      <td>harry potter and the order of the phoenix harr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>eng</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>11/1/03</td>
      <td>Scholastic</td>
      <td>harry potter and the chamber of secrets harry ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>eng</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>5/1/04</td>
      <td>Scholastic Inc.</td>
      <td>harry potter and the prisoner of azkaban harry...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>eng</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>9/13/04</td>
      <td>Scholastic</td>
      <td>harry potter boxed set books 15 harry potter 15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11118</th>
      <td>45631</td>
      <td>Expelled from Eden: A William T. Vollmann Reader</td>
      <td>William T. Vollmann/Larry McCaffery/Michael He...</td>
      <td>4.06</td>
      <td>eng</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>12/21/04</td>
      <td>Da Capo Press</td>
      <td>expelled from eden a william t vollmann reader</td>
    </tr>
    <tr>
      <th>11119</th>
      <td>45633</td>
      <td>You Bright and Risen Angels</td>
      <td>William T. Vollmann</td>
      <td>4.08</td>
      <td>eng</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>12/1/88</td>
      <td>Penguin Books</td>
      <td>you bright and risen angels</td>
    </tr>
    <tr>
      <th>11120</th>
      <td>45634</td>
      <td>The Ice-Shirt (Seven Dreams #1)</td>
      <td>William T. Vollmann</td>
      <td>3.96</td>
      <td>eng</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>8/1/93</td>
      <td>Penguin Books</td>
      <td>the iceshirt seven dreams 1</td>
    </tr>
    <tr>
      <th>11121</th>
      <td>45639</td>
      <td>Poor People</td>
      <td>William T. Vollmann</td>
      <td>3.72</td>
      <td>eng</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2/27/07</td>
      <td>Ecco</td>
      <td>poor people</td>
    </tr>
    <tr>
      <th>11122</th>
      <td>45641</td>
      <td>Las aventuras de Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>3.91</td>
      <td>spa</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>5/28/06</td>
      <td>Edimat Libros</td>
      <td>las aventuras de tom sawyer</td>
    </tr>
  </tbody>
</table>
<p>11120 rows × 11 columns</p>
</div>




```python
# remove null titles
book_data = book_data[book_data["mod_title"].str.len() > 0]
```


```python
book_data.shape
# the above process removed 3 books
```




    (11120, 11)



# Building the search engine
This search engine will match the title of a book you input and also find books with similar titles to display. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(book_data["mod_title"])
```


```python
# turn search query into vector and match it against the tfidf matrix and then compare (using a function)
from sklearn.metrics.pairwise import cosine_similarity

def search(query, vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    # find the 10 largest similarities
    indices = np.argpartition(similarity, -10)[-10:]

    # index the titles
    results = book_data.iloc[indices]

    # sort results on the highest number of ratings
    results = results.sort_values("ratings_count", ascending = False)
    return results.head(5)
```


```python
# run the function with a book title
search("the alchemist", vectorizer)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
      <th>mod_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>284</th>
      <td>865</td>
      <td>The Alchemist</td>
      <td>Paulo Coelho/Alan R. Clarke/Özdemir İnce</td>
      <td>3.86</td>
      <td>eng</td>
      <td>197</td>
      <td>1631221</td>
      <td>55843</td>
      <td>5/1/93</td>
      <td>HarperCollins</td>
      <td>the alchemist</td>
    </tr>
    <tr>
      <th>288</th>
      <td>870</td>
      <td>Fullmetal Alchemist  Vol. 1 (Fullmetal Alchemi...</td>
      <td>Hiromu Arakawa/Akira Watanabe</td>
      <td>4.50</td>
      <td>eng</td>
      <td>192</td>
      <td>111091</td>
      <td>1427</td>
      <td>5/3/05</td>
      <td>VIZ Media LLC</td>
      <td>fullmetal alchemist vol 1 fullmetal alchemist 1</td>
    </tr>
    <tr>
      <th>287</th>
      <td>869</td>
      <td>Fullmetal Alchemist  Vol. 8 (Fullmetal Alchemi...</td>
      <td>Hiromu Arakawa/Akira Watanabe</td>
      <td>4.57</td>
      <td>eng</td>
      <td>192</td>
      <td>11451</td>
      <td>161</td>
      <td>7/18/06</td>
      <td>VIZ Media LLC</td>
      <td>fullmetal alchemist vol 8 fullmetal alchemist 8</td>
    </tr>
    <tr>
      <th>285</th>
      <td>866</td>
      <td>Fullmetal Alchemist  Vol. 9 (Fullmetal Alchemi...</td>
      <td>Hiromu Arakawa/Akira Watanabe</td>
      <td>4.57</td>
      <td>eng</td>
      <td>192</td>
      <td>9013</td>
      <td>153</td>
      <td>9/19/06</td>
      <td>VIZ Media LLC</td>
      <td>fullmetal alchemist vol 9 fullmetal alchemist 9</td>
    </tr>
    <tr>
      <th>6978</th>
      <td>26425</td>
      <td>Fullmetal Alchemist: The Abducted Alchemist (F...</td>
      <td>Makoto Inoue/Hiromu Arakawa/Alexander O. Smith...</td>
      <td>4.57</td>
      <td>eng</td>
      <td>240</td>
      <td>2779</td>
      <td>19</td>
      <td>1/10/06</td>
      <td>VIZ Media LLC</td>
      <td>fullmetal alchemist the abducted alchemist ful...</td>
    </tr>
  </tbody>
</table>
</div>



# Creating a list of liked books
In this section, I will be using my search engine from above to query the books that I read (titles from my personal Goodreads account; https://www.goodreads.com/user/show/50112085-star) and create a list of their book ids if the query exists in the data.


```python
# book ids of 7 of the books I like
liked_books = ["37415", "10210", "27451", "7613", "3636", "22188","865"]
```


```python
# Display the rows of books from my liked books list by their book ID
book_data.loc[book_data["bookID"].isin(["37415", "10210", "27451", "7613", "3636", "22188","865"])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
      <th>mod_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>284</th>
      <td>865</td>
      <td>The Alchemist</td>
      <td>Paulo Coelho/Alan R. Clarke/Özdemir İnce</td>
      <td>3.86</td>
      <td>eng</td>
      <td>197</td>
      <td>1631221</td>
      <td>55843</td>
      <td>5/1/93</td>
      <td>HarperCollins</td>
      <td>the alchemist</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>3636</td>
      <td>The Giver (The Giver  #1)</td>
      <td>Lois Lowry</td>
      <td>4.13</td>
      <td>eng</td>
      <td>208</td>
      <td>1585589</td>
      <td>56604</td>
      <td>1/24/06</td>
      <td>Ember</td>
      <td>the giver the giver 1</td>
    </tr>
    <tr>
      <th>2114</th>
      <td>7613</td>
      <td>Animal Farm</td>
      <td>George Orwell/Boris Grabnar/Peter Škerl</td>
      <td>3.93</td>
      <td>eng</td>
      <td>122</td>
      <td>2111750</td>
      <td>29677</td>
      <td>5/6/03</td>
      <td>NAL</td>
      <td>animal farm</td>
    </tr>
    <tr>
      <th>2764</th>
      <td>10210</td>
      <td>Jane Eyre</td>
      <td>Charlotte Brontë/Michael Mason</td>
      <td>4.12</td>
      <td>eng</td>
      <td>532</td>
      <td>1409369</td>
      <td>27884</td>
      <td>2/4/03</td>
      <td>Penguin</td>
      <td>jane eyre</td>
    </tr>
    <tr>
      <th>5887</th>
      <td>22188</td>
      <td>Gossip Girl (Gossip Girl  #1)</td>
      <td>Cecily von Ziegesar</td>
      <td>3.52</td>
      <td>eng</td>
      <td>224</td>
      <td>54400</td>
      <td>2271</td>
      <td>4/1/02</td>
      <td>Little  Brown and Company</td>
      <td>gossip girl gossip girl 1</td>
    </tr>
    <tr>
      <th>7161</th>
      <td>27451</td>
      <td>The Great Gatsby</td>
      <td>F. Scott Fitzgerald/Matthew J. Bruccoli</td>
      <td>3.91</td>
      <td>eng</td>
      <td>216</td>
      <td>9844</td>
      <td>1050</td>
      <td>6/1/95</td>
      <td>Scribner</td>
      <td>the great gatsby</td>
    </tr>
    <tr>
      <th>9427</th>
      <td>37415</td>
      <td>Their Eyes Were Watching God</td>
      <td>Zora Neale Hurston</td>
      <td>3.91</td>
      <td>eng</td>
      <td>219</td>
      <td>220309</td>
      <td>9536</td>
      <td>5/30/06</td>
      <td>Amistad</td>
      <td>their eyes were watching god</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
