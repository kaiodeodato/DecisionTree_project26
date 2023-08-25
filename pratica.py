# importando módulos necessários
# DecisionTreeClassifier classe que constroi classificador de árvore de decisão
from pyspark.ml.classification import DecisionTreeClassifier
# VectorAssembler transforma colunas em vetor, que os algoritimos de ML conseguem ler
# StringIndexer é utilizado para transformar valores categóricos em numéricos
# IndexToString é utilizado para transformar valores numéricos em categóricos
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
# Importa a classe SparkSession para criar uma sessão do Spark.
from pyspark.sql import SparkSession

# Inicializa uma sessão spark com o nome "app_model"
spark = SparkSession.builder.appName("app_model").getOrCreate()
# Lê um arquivo csv em um dataframe chamado 'df', considerando a primeira linha como cabeçalho
df = spark.read.option("header", "true").csv("stocks_2021.csv")
# Filtra o DataFrame para manter apenas as linhas em que a coluna 'ticker' corresponde ao valor 'BA', 
# resultando em um DataFrame contendo apenas dados da Boeing.
stock1 = 'BA'
df_stock1 = df.filter(df.ticker == stock1)
# Converte as colunas para o tipo de dado float.
df = df.withColumn('open', df.open.cast('float'))
df = df.withColumn('low', df.low.cast('float'))
df = df.withColumn('close', df.close.cast('float'))
# Cria um VectorAssembler que combina as colunas 'open', 'low' e 'close' em uma única coluna de 
# características chamada 'features'.
va = VectorAssembler(inputCols=['open', 'low', 'close'], outputCol='features')
# Aplica a transformação VectorAssembler ao DataFrame, criando um novo DataFrame va_df com a coluna 'features'.
va_df = va.transform(df)

# transformo os valores em ticker em valures numéricos e os coloco em label
indexer = StringIndexer(inputCol='ticker', outputCol='label')
# treino o modelo com os novos valores de label
indexer_model = indexer.fit(va_df)
indexed_df = indexer_model.transform(va_df)

# crio um onjeto DecisionTreeClassifier usando a coluna com os nomes já decodificados de meus 'tickers'
# e minha coluna combinada 'features'
dtc = DecisionTreeClassifier(featuresCol='features', labelCol='label')

# divido o dataframe e duas porções 'train' e 'test', 20% e 80%
(train, test) = indexed_df.randomSplit([0.2, 0.8])

# Treina o modelo com base no conjunto de dados de treinamento
dtc_model = dtc.fit(train)
# Aplicar o modelo treinado a outra porção, a parte do dataframe de teste
pred = dtc_model.transform(test)
# criando transformador usando IndexToString para reverter a transformação feita com StringIndexer
# pego os dados da coluna 'label' e ponho na nova coluna 'ticker_pred'
converter = IndexToString(inputCol="label", outputCol="ticker_pred", labels=indexer_model.labels)
# Utilizo o conversor para aplicar a transformação de conversão dos valores numéricos da coluna 'label' de volta
# aos valores categóricos da coluna 'ticker'
# isso é feito no datafram 'pred', que contém as previsões feitas pelo modelo de árvore de decisão
pred_with_ticker = converter.transform(pred)
# juntando o datagram filtrado original df_stock1 com o dataframe resultante da aplicação do modelo
joined_df = pred_with_ticker.join(df_stock1, on='ticker', how='inner')
# exibir resultados no console
joined_df.show()

