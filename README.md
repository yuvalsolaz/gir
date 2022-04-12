# gir
Geographical Information Retrieval

dataset:

Twitter spritzer: 
A simple collection of JSON grabbed from the general twitter stream, for the purposes of research, history, testing and memory. This is the "Spritzer" version,  the most light and shallow of Twitter grabs. Unfortunately, we do not currently have access to the Sprinkler or Garden Hose versions of the stream. https://archive.org/details/twitterstream?&sort=-week&page=2
The site contains one zip file (~2 giga) for each month - around 2M tweets per day
Only 1 percent of the tweets contains location 

The load_twitter script open all files in a given directory
Extract all location tweets and save to one csv file