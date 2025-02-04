from flask import Flask, request, render_template, jsonify
from src.Creditcardfaultdetection.pipelines.prediction_pipeline import custom_data, predictPipeline

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template("form.html")
    
        else: 
            data = custom_data(
                LIMIT_BAL=  float(request.form.get('limit_balance')),
                PAY_1 = str(request.form.get('pay_1')),
                PAY_2 = str(request.form.get('pay_2')),
                PAY_3 = str(request.form.get('pay_3')),
                PAY_4 = str(request.form.get('pay_4')),
                PAY_5 = str(request.form.get('pay_5')),
                PAY_6 = str(request.form.get('pay_6'))
            )
            
            
            df=data.get_data_as_df()

            pred=predictPipeline()
            # Get prediction result from the pipeline
            result =pred.predict(df)
            
            # Return the prediction result
            return render_template("result.html",final_result=result)
    
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)

