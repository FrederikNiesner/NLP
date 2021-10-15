# %%

# import python packages
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Load english NLTK model with stopwords 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 
sw = stopwords.words('english')

comments = pd.read_excel(r'/Users/fred/OneDrive - Adobe/Data/NLP_sentiment/Feedback Surveys/Global Utilization Report Satisfaction Survey.xlsx') 

# FUNCTIONS


def clean_text(text):
    """Define a general function to apply standard text cleaning measures on text data.
    Remove all capital letters, punctuations, emojis, links, etc. Basically, removing all that is not words or numbers.
    Tokenize the data into words, which means breaking up every comment into a group of individual words.

    Args:
        text (string): Input raw text. 

    Returns:
        string: Pre-processed and tokenized text.
    """

    # Remove non-words
    text = text.lower()
    text = re.sub('@', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove Stopwords
    text = [w for w in text if w not in sw]

    return text


def lem(text):
    """[summary]

    Args:
        text (string): Pre-processed input text from survey data.

    Returns:
        list: Lemmatized text
    """    
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [lemmatizer.lemmatize(t, 'v') for t in text]
    return text


# TEXT PREPROCESSING

# Create a dataframe for each of the text columns and drop empty fields
df_improvements = comments[['Which aspect of the report needs the most improvement?']].dropna()
df_feedback = comments[['Do you have any other feedback about this report?']].dropna()

# Starting with general feedback for this notebook
comments = df_feedback

# reset index and rename the column to 'text'
comments.reset_index(drop=True, inplace=True)
comments.columns = ['text']

# Apply clean_text function
comments['text'] = comments['text'].apply(lambda x: clean_text(x))

# Apply Lemmatization function
lemmatizer = WordNetLemmatizer()
comments['text'] = comments['text'].apply(lambda x: lem(x))  # check here one more time

# %%


comments[:4]


# %%


# # Analyzing the data
# 
# ## Word Frequency
# Let’s begin by looking at the word frequency, i.e. what words are repeated most often in the comments, using the FreqDist function from nltk.

# From lists of comments to a single list containing all words      
all_words=[]        
for i in range(len(comments)):
    all_words = all_words + comments['text'][i]

# Get word frequency        
nlp_words = nltk.FreqDist(all_words)
plot1 = nlp_words.plot(20, color='salmon', title='Word Frequency')

# %% [markdown]
# That does not give us much idea about how people feel about the video. Still, here are a few noteworthy mentions:
# 
# Let’s look for the most frequent bigrams, which means the most frequent pair of words that are next to each other in a comment.
# 

# %%
# Bigrams
bigrm = list(nltk.bigrams(all_words))
words_2 = nltk.FreqDist(bigrm)
words_2.plot(20, color='salmon', title='Bigram Frequency')

# %% [markdown]
# Now we are starting to see some more interesting stuff. The discussion circulates a lot around the friday live q&a session. 
# 
# Here is the plot for the most popular trigrams basically underlining what the bigrams already indicate. 

# %%
# Let's shift gears to "Trigrams"
trigrm = list(nltk.trigrams(all_words))
words_3 = nltk.FreqDist(trigrm)
words_3.plot(20, color='salmon', title='Bigram Frequency')

# %% [markdown]
# Not a lot of new information. 
# 
# But can we trust these results? 
# 
# #### If we take a closer look at the data it seems the video author is moderating A LOT in the comment section. Therefore, a lot of comments will just express his opinion and the dataset is BIASED ...
# 
# #### What if we exclude the author comments? Will this provide another result purely from the user's perspective?

# %%
# Get the dataset ready
comments = pd.read_json(r'/Users/fred/OneDrive - Adobe/Data/NLP_sentiment/Youtube_Comments/AGrl-H87pRU_comments.json', lines=True) 
comments = comments[comments.author != "Avi Singh - PowerBIPro"] # Excluding Avi's (the author) comments
comments = comments.reset_index(drop=True)
comments = comments.drop(['cid', 'time', 'author', 'channel', 'votes', 'photo', 'heart'], axis=1)

# Apply the cleaning and lemmatization functions
comments['text'] = comments['text'].apply(lambda x: clean_text(x))
comments['text'] = comments['text'].apply(lambda x: lem(x))

# From lists of comments to a single list containing all words      
all_words=[]        
for i in range(len(comments)):
    all_words = all_words + comments['text'][i]

# Get word frequency        
nlp_words = nltk.FreqDist(all_words)
plot1 = nlp_words.plot(20, color='blue', title='Word Frequency')

# Bigrams
bigrm = list(nltk.bigrams(all_words))
words_2 = nltk.FreqDist(bigrm)
words_2.plot(20, color='blue', title='Bigram Frequency')

# %% [markdown]
# Well the discussion did quite shift a lot. 
# 
# * No more talking about the PowerBI QA session live event on Friday
# * Fun fact. Illuminati are interested in PowerBI, too ;-)
# %% [markdown]
# # Sentiment Analysis
# 
# ### But first. A bit of theory. 
# 
# Let's ask wikipedia one last time: 
# 
# *Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.*
# 
# The sentiment property below returns a named tuple of the form Sentiment(polarity, subjectivity). 
# 
# * The polarity score is a float within the range [-1.0=negative to 1.0=positive]
# * The subjectivity is a float within the range [0.0 to 1.0] where 0.0 is very objective and 1.0 is very subjective.
# 
# 

# %%
# Get sentiment for comment data

comments['text'] = [str(thing) for thing in comments['text']]
sentiment = []
for i in range(len(comments)):
    blob = TextBlob(comments['text'][i])
    for sentence in blob.sentences:
        sentiment.append(sentence.sentiment.polarity)

comments['sentiment'] = sentiment


# %%
# Let's take a look at the our dataset with comments.head()

comments.sort_values(by=['sentiment'], ascending=False)

# %% [markdown]
# Interesting! It seems obvious that the best sentiment comes from comments like "You're awesome man" with a sentiment score of 1.
# 
# Well! At the other end with a negative sentiment score it seems someone people did indeed complain. 
# E.g. about the PowerBI pricing model. Fortunatelly we don't need to know how expensive PowerBI licenses are ;-) 
# 

# %%

# Let's plot the sentiment as histogram
comments['sentiment'].plot.hist(color='salmon', title='Comments Polarity')

# %% [markdown]
# ## Now! How could a use case for our team look like.
# 
# Let's imagine some user feedback on our reports as below.

# %%
### BEFORE YOU CONTINUE READING: 
### QUESTION: What do you think is the sentiment and subjectivity score for the below feedback phrases? ##

PSAI_feedback = pd.DataFrame({'feedback': [
 "PS Analytics & Insights team are the worst",   
 "PS Analytics & Insights team really doesn't perform well these days",   
 "PS Analytics & Insights team",   
 "PS Analytics & Insights team is pretty awesome",   
 "PS Analytics & Insights team is awesome",   
]})






# %%
PSAI_feedback['feedback'] = [str(thing) for thing in PSAI_feedback['feedback']] # actually not necessary here.

sentiment = []
subjectivity = []

for i in range(len(PSAI_feedback)):
    blob = TextBlob(PSAI_feedback['feedback'][i])
    for sentence in blob.sentences:
        sentiment.append(round(sentence.sentiment.polarity, 1))
        subjectivity.append(round(sentence.sentiment.subjectivity, 1))

PSAI_feedback["sentiment"] = sentiment
PSAI_feedback["subjectivity"] = subjectivity
PSAI_feedback.head()


# %%
# Take a closer look at subjectivity. 
# This is an important metric. For example the first sentence appears as very negative feedback. But it is also very subjective, meaning just expressing the opinion of one person and without for detail or prove!! Thus, subjective...

# %% [markdown]
# ## Let's apply some REAL user feedback from previous survey. 
# 
# Below I used some samples from a GU report feedback survey in 2019.
# 
# Survey question: What can we do to make your experience better?
# 

# %%
survey_feedback = pd.DataFrame({'feedback': [
 "Awesome",   
 "Its just PERFECT",   
 "The ability to sort without looking at Plan, maybe further make it easier by org?", 
 "This session was covering basic which most of the people managed to enable on their own. We would love to have a session to know all extra features and benefits from Power BI report.",   
 "indicate how the performance achieved and confirmation of variable achievement",   
 "Insert the soft booking in the forecast, in order to have a more complete view of the forecast.",   
 "Nothing, thank you for providing an excellent report.",
 "more hands on sessions"
]})

# This time we need to quickly apply our clean function from above since we have long text with punctuation. 
# Otherwise we would get a result for each sentence in 1 feedback, but we want the sentiment for all sentences in 1 feedback)
survey_feedback['feedback'] = survey_feedback['feedback'].apply(lambda x: clean_text(x))
survey_feedback['feedback'] = [str(thing) for thing in survey_feedback['feedback']]



# Let's run a sentiment analysis with default TextBlob
sentiment = []
subjectivity = []

for i in range(len(survey_feedback)):
    blob = TextBlob(survey_feedback['feedback'][i])
    for sentence in blob.sentences:
        sentiment.append(round(sentence.sentiment.polarity, 1))
        subjectivity.append(round(sentence.sentiment.subjectivity, 1))

"""

# Uncomment to leverage NaiveBayes classifier. This could take a while. 
sentiment = []

for i in range(len(survey_feedback)):
    blob = TextBlob(survey_feedback['feedback'][i], analyzer=NaiveBayesAnalyzer())
    for sentence in blob.sentences:
        sentiment.append(sentence.sentiment)
"""

survey_feedback["subjectivity"] = subjectivity
survey_feedback["sentiment"] = sentiment
survey_feedback.head()

# %% [markdown]
# ##### Again the results are interesting. 
# 
# * While the first two comments are "nice to have" they are super subjective and therefore do not deliver a tangible result to derive actions from (except to be happy maybe... ;-) )
# * The 3rd sentence actually was a question -> so neither positive/negative nor subjective/objective
# * Finally, the 4th sentence implies that the training session was just providing basic content that the users already knew.
# * Thus, our classifier corretly tells us: This seems to be valid feedback ("pretty objective") and the user wasn't very happy (sentiment score 0.2)
# %% [markdown]
# # Whats'next? 
# 
# * If you made it all through the notebook until this point, why don't you let me know what you think of this approach. Just reach out!!!! 
# 
# Fred ;-)
# 

# %%



# %%



# %%



