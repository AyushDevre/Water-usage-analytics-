from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum

# Start Spark
spark = SparkSession.builder \
    .appName("WaterUsageAnalysis") \
    .getOrCreate()

# Load data
df = spark.read.csv("data/data.csv", header=True, inferSchema=True)

# Show data
print("Original Data:")
df.show(5)

# 🔥 Aggregation 1: Total water usage per industry
water_by_industry = df.groupBy("industry").agg(
    sum("water_usage").alias("total_water_usage")
)

# 🔥 Aggregation 2: Average efficiency
efficiency_df = df.withColumn(
    "efficiency",
    col("water_usage") / col("production_units")
).groupBy("industry").agg(
    avg("efficiency").alias("avg_efficiency")
)

# Show results
print("Water Usage by Industry:")
water_by_industry.show()

print("Efficiency by Industry:")
efficiency_df.show()

# Save processed data
water_by_industry.toPandas().to_csv("data/water_by_industry.csv", index=False)
efficiency_df.toPandas().to_csv("data/efficiency.csv", index=False)

spark.stop()