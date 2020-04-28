from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sentences = open('Your_File_Name.txt')

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, lowercase=False, ngram_range=(3,4)) # ngram_range (min value of ngrams, max value of ngrams)

tfidf.fit_transform(sentences)

print(tfidf.vocabulary_)

#Method using CountVectorizer

#vectorizer = CountVectorizer(min_df=10, lowercase=False, ngram_range=(2,3))
#vectorizer.fit(sentences)
#print(vectorizer.vocabulary_)

