from src.Creditcardfaultdetection.pipelines.prediction_pipeline import custom_data,predictPipeline

data=custom_data(55000,'pay duly','pay duly','pay duly','pay duly','pay duly','pay duly')
df=data.get_data_as_df()
print(df)
pred=predictPipeline()
output=pred.predict(df)

print(output)
