# Project Overview
In this project, I created a search engine in order to query book titles, gather their unique book IDs, and display all of the books' information. This application is useful for people who have a list of books (favorite books, to-be-read, popular books etc.) and would like to view all the data from those books at once

## Step 1
- The first thing I did was cleaned the data using OpenRefine (https://openrefine.org/download.html)
- In this step, I 

## Step 2
- I imported the clean dataset to Jupyter noebook and cleaned the data more (for the search engine to work) by creating a new column for book titles that removed non-alphanumeric characters, transforming the titles to lowercase, removing multiple spaces in a row if they exists, and removing rows with null titles.

## Step 3
- I built the search engine by using the scikit learn package from python specifically the Tfidf Vectorizer to analyze term frequency and document frequency in a query and find the 10 largest similarities
- I then ordered the results from highest to lowest number book ratings because the higher the number of ratings, the most likely the correct version of the book you're looking for is in the top of the results.

## Step 4
- I created a list of book IDs by using the search engine to query some of my favorite books

## Step 5
- I displayed all of the information from the list of my favorite books
