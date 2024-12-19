from flask import Flask, render_template, request,url_for,redirect
import numpy as np
import pickle


app=Flask(__name__)

model=pickle.load(open('diabetes.pkl','rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]

    arr=[np.array(int_features)]
    pred=model.predict(arr)
    return render_template('home.html',p=format(pred))


if __name__ == "__main__":
    app.run(debug=True)

