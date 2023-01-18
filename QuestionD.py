from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import desc
from nltk.corpus import stopwords

# D) Based on b, obtain the top 20 most frequently occurred words
# except the stop words such as “a”, “the”, “this” and so on.

# Create a SparkSession
spark = SparkSession.builder.appName("Word Count").getOrCreate()

# Read the text file into a DataFrame
df = spark.read.text("Trump_Tweet")

# Split the text into words, remove punctuations
words = df.select(explode(split(regexp_replace(lower(col("value")), "[^a-zA-Z\\s]", ""), " ")).alias("word"))
# nltk.download('stopwords')

# Get a list of stop words

stop_words = set(stopwords.words('english'))

# # add to the stop words list the same word with the first letter in upper case, to cover all options.
# new_stop_words = set()
# for word in stop_words:
#     new_word = word[0].upper() + word[1:]
#     new_stop_words.add(new_word)
#
# # add the new words to the original stop words set
# stop_words = stop_words.union(new_stop_words)
#
# print("Stop words are:\n",stop_words)

# trim the word and filter out empty words
words = words.filter(length(trim(col("word"))) > 0)

# filter out words that are not in stop words
words = words.filter(~ words.word.isin(stop_words))

# Count the number of occurrences of each word, and sort by descending count.
word_count = words.groupBy("word").count()
word_count = word_count.sort(desc("count"))
# Show only top 20
word_count.show(20)


