from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("app_model").getOrCreate()

df = spark.read.option("header", "true").csv("stocks_2021.csv")

stock1 = 'BA'
df_stock1 = df.filter(df.ticker == stock1)

df = df.withColumn('open', df.open.cast('float'))
df = df.withColumn('low', df.low.cast('float'))
df = df.withColumn('close', df.close.cast('float'))

va = VectorAssembler(inputCols=['open', 'low', 'close'], outputCol='features')

va_df = va.transform(df)

indexer = StringIndexer(inputCol='ticker', outputCol='label')

indexer_model = indexer.fit(va_df)
indexed_df = indexer_model.transform(va_df)

dtc = DecisionTreeClassifier(featuresCol='features', labelCol='label')

(train, test) = indexed_df.randomSplit([0.2, 0.8])

dtc_model = dtc.fit(train)

pred = dtc_model.transform(test)

converter = IndexToString(inputCol="label", outputCol="ticker_pred", labels=indexer_model.labels)

pred_with_ticker = converter.transform(pred)

joined_df = pred_with_ticker.join(df_stock1, on='ticker', how='inner')

joined_df.show()

