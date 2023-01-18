from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# B) Count the number of occurrences of each word in the file. The count should be
# case-insensitive and ignore punctuations.

# Create a SparkSession
spark = SparkSession.builder.appName("Word Count").getOrCreate()

# Read the text file into a DataFrame
df = spark.read.text("Trump_Tweet")

# Split the text into words, remove punctuations
words = df.select(explode(split(regexp_replace(lower(col("value")), "[^a-zA-Z\\s]", ""), " ")).alias("word"))

# trim the word and filter out empty words
words = words.filter(length(trim(col("word"))) > 0)

# Count the number of occurrences of each word, and sort by count occurrence
word_count = words.groupBy("word").count()
word_count = word_count.sort(desc("count"))

# Show the result
results_number = 30
# to show all results, replace with: truncate=False
word_count.show(results_number)

