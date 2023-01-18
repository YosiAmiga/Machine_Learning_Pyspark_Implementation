from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.functions import explode, split


# C) Based on b, count the number of occurrences of 2-gram in the file. A 2-gram is a
# sequence that contains the adjacent two words. For example, the 2-gram in the
# sentence “I like to eat pizza” is “I like”, “like to”, “to eat” and “eat pizza”.

# Create a SparkSession
spark = SparkSession.builder.appName("Bi-Gram Count").getOrCreate()

# Read the text file into a DataFrame
df = spark.read.text("Trump_Tweet")

# Split the text into words and remove punctuations
words = df.select(split(regexp_replace(lower(col("value")), "[^a-zA-Z\\s]", ""), " ").alias("line"),
                  monotonically_increasing_id().alias("line_id"))

# break each array of words of each line into a df with columns: line id, pos, word
words = words.select("line_id", posexplode("line").alias("pos", "word"))

# trim the word and filter out empty words
words = words.filter(length(trim(col("word"))) > 0)

# Create a window function to define the 2-grams
w = Window().partitionBy(col("line_id")).orderBy(monotonically_increasing_id())

# Add a new column to the DataFrame containing the next word in the 2-gram
df_2gram = words.withColumn("next_word", lead("word").over(w)).filter(col("next_word").isNotNull())

# Count the number of occurrences of each 2-gram, and sort them by descending occurrences
df_2gram = df_2gram.withColumn("2-gram",concat(col("word"), lit(" "), col("next_word"))).groupBy("2-gram").count()

df_2gram = df_2gram.sort(desc("count"))

# Show the result
results_number = 30

# to show all results, replace with: truncate=False
df_2gram.show(results_number)




