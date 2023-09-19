from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction import CustomData, PredictingPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','Post'])
def predict_data():
    if request.method=='Get':
        return render_template('form.html')
    
    else:
        data = CustomData(carat = float(request.form.get('carat')),
        depth = float(request.form.get('depth')),
        table = float(request.form.get('table')),
        x = float(request.form.get('x')),
        y = float(request.form.get('y')),
        z = float(request.form.get('z')),
        cut = request.form.get('cut'),
        color = request.form.get('color'),
        clarity = request.form.get('clarity'))

        final_data = data.get_data()
        predict_pipeline = PredictingPipeline()
        pred = predict_pipeline.predict(final_data)

        results = round(pred[0],2)
        return render_template('form.html', final_result = results)

if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)