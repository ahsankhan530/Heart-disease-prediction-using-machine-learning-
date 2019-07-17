from flask import Flask,request

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

df=pd.read_csv('heart.csv')
Y=df.target.values
x_data=df.drop(['target'],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=0)
#transpose matrices
x_train = x_train
y_train = y_train
x_test = x_test
y_test = y_test
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model=nb.fit(x_train, y_train)
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/add',methods=['POST'])
def add():
    a=request.form
    age=a['age']
    sex=a['sex']
    cp=a['cp']
    trestbps=a['trestbps']
    chol=a['chol']
    fbs=a['fbs']
    restecg=a['restecg']
    
    thalach=a['thalach']
    exang=a['exang']
    oldpeak=a['oldpeak']
    slope=a['slope']
    ca=a['ca']
    thal=a['thal']
    print('aaaa',age)
    age=int(age)
    sex=int(sex)
    cp=int(cp)
    trestbps=int(trestbps)
    chol=int(chol)
    fbs=int(fbs)
    restecg=int(restecg)
    ca=int(ca)
    thalach=int(thalach)
    exang=int(exang)
    oldpeak=int(oldpeak)
    slope=int(slope)
    thal=int(thal)
    r=[[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]
    print('rrr',r)
    aq=model.predict(r)
    print('rrrrrr',aq.dtype)
    #abs= np.array([[a['age'],a['sex'],a['cp'],a['trestbps'],a['chol'],a['fbs'],a['restecg'],a['thalach'],a['exang'],a['oldpeak'],a['slope'],a['ca'],a['thal']) 
    # print('sssssssss',abs.dtype)


    data=[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    # abs=np.array([abs])
    print("abs",data)
    
    # a=np.array([[a]],dtype='int32')
    
    # print(type(a))
    rediction=model.predict(data)
    # print('ssss',rediction.dtype)
    if rediction==1:
        return render_template('a.html')
    else:
        return render_template('b.html')
    
# @app.route('/aaa',methods=['POST'])
# def aaa():
#     data=request.get_json(force=True)
#     pred=model.predict([[data['age'],data['sex'],data['cp'],data['trestbps'],data['chol'],data['fbs'],data['restecg'],data['thalach'],
#     data['exang'],data['oldpeak'],data['slope'],data['ca'],data['thal']]])
#     output=[pred[0]]
#     print (pred)
#     if pred== 1:
#         return 'yes you have'
#     else:
#         return 'no'
#     return render_template('index.html',out=pred)
if __name__ == "__main__":
    app.run(debug=True)

