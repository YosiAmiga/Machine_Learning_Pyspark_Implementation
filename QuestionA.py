from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# A) Count the number of occurrences of each alphabetic character in a file.
# The count for each letter should be case-insensitive
# Include both upper-case and lower-case versions of the letter, and ignore non-alphabetic characters


# Create a SparkSession
spark = SparkSession.builder.appName("Character Count").getOrCreate()

# Read the text file into a DataFrame
df = spark.read.text("Trump_Tweet")

# Count the number of occurrences of each alphabetic character in the file, and sort by count occurrence
char_count = df.select(explode(split(lower(col("value")), "")).alias("char")) \
    .filter(col("char").rlike("[A-Za-z]")) \
    .groupBy("char").count()

char_count = char_count.sort(desc("count"))

# Show the result
results_number = 30
# to show all results, replace with: truncate=False
char_count.show(results_number)

