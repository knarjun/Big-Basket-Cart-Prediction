import numpy as np
import pandas as pd
from apyori import apriori
from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    product_name = request.form.get('product_name', '').strip()
    if uploaded_file.filename != '':
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
        dataset = pd.read_csv(file_path, header=None)
        transactions = []
        for i in range(len(dataset)):
            transactions.append([str(dataset.values[i, j]) for j in range(len(dataset.columns))])

        basket_intelligence = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_length=2, min_lift=3)
        results = list(basket_intelligence)

        def inspect(results):
            product1 = [tuple(result[2][0][0])[0] for result in results]
            product2 = [tuple(result[2][0][1])[0] for result in results]
            supports = [result[1] for result in results]
            confidences = [result[2][0][2] for result in results]
            lifts = [result[2][0][3] for result in results]
            return list(zip(product1, product2, supports, confidences, lifts))

        DataFrame_intelligence = pd.DataFrame(inspect(results), columns=['product1', 'product2', 'support', 'confidence', 'lift'])

        top_10_lift = DataFrame_intelligence.nlargest(n=10, columns='lift')

        if product_name:
            picked_more = DataFrame_intelligence[(DataFrame_intelligence['product1'] == product_name) | (DataFrame_intelligence['product2'] == product_name)]
            if not picked_more.empty:
                return render_template('result.html', tables=[top_10_lift.to_html(classes='data', header="true")], picked_more=picked_more.to_html(classes='data', header="true"), product_name=product_name)
        
        return render_template('result.html', tables=[top_10_lift.to_html(classes='data', header="true")], product_name=product_name)
    else:
        return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
