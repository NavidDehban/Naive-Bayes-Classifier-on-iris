import numpy as np
import pandas as pd
df = pd.read_csv(r"E:\university\term 6\ML\home works\hw1\HW1\Iris.csv")
def topic_detector(df):
    topic = []
    for i in df:
        topic.append(i)
    return topic 
def class_detector(df):
    topic = topic_detector(df)
    classes = df[topic[4]].tolist()
    classes = set(classes)
    classes = list(classes)
    return classes
def data_divider(df):
    msk = np.random.rand(len(df)) < 0.75
    train = df[msk]
    test = df[~msk]
    return [test,train]
def seprator(df):
    classes = class_detector(df)
    topic = topic_detector(df)
    df1 = df.loc[df[topic[4]]==classes[0]]
    df2 = df.loc[df[topic[4]]==classes[1]]
    df3 = df.loc[df[topic[4]]==classes[2]]
    return [df1,df2,df3]
def mean(df):
    topic = topic_detector(df)
    m1 = df[topic[0]].mean()
    m2 = df[topic[1]].mean()
    m3 = df[topic[2]].mean()
    m4 = df[topic[3]].mean()
    return np.array([[m1],[m2],[m3],[m4]])
def variance_(df):
    topic = topic_detector(df)
    m1 = df[topic[0]].var()
    m2 = df[topic[1]].var()
    m3 = df[topic[2]].var()
    m4 = df[topic[3]].var()
    return np.array([[m1],[m2],[m3],[m4]])
def covariance(df):
    topics = topic_detector(df)
    var = pd.concat([df[topics[0]],df[topics[1]],df[topics[2]],df[topics[3]]], axis=1)
    c = var.cov()
    c = c.to_numpy(dtype = 'float')
    return c
def g(x,m,c,p):
    d = 4
    A = np.subtract(x,m)
    B = np.linalg.inv(c)
    C = d*np.log(2*np.pi)/2
    D = np.log(np.linalg.det(c))/2
    E = np.log(p)
    atb = np.dot(A.transpose(),B)
    g = -0.5*np.dot(atb,A) -C -D +E
    return g[0][0]
def prior(df):
    topic = topic_detector(df)
    classes = class_detector(df)
    p = []
    for i in range(3):
        p.append(len(df.loc[df[topic[4]]==classes[0]][topic[4]].tolist()))
    s = sum(p)
    for i in range(3):
        p[i] = p[i]/s
    return p
def cmp1(x,m,c,p,classes):
    m1,m2,m3 = m
    c1,c2,c3 = c
    p1,p2,p3 = p
    g1 = g(x,m1,c1,p1)
    g2 = g(x,m2,c2,p2)
    g3 = g(x,m3,c3,p3)
    compare = [[0,g1],[1,g2],[2,g3]]
    compare = sorted(compare, key=lambda x: x[1],reverse= True)
    return classes[compare[0][0]]
def optimal_bayes(df):
    predict = []
    classes = class_detector(df)
    topic = topic_detector(df)
    test,train = data_divider(df)
    df1,df2,df3 = seprator(train)
    m1 = mean(df1)
    c1 = covariance(df1)
    m2 = mean(df2)
    c2 = covariance(df2)
    m3 = mean(df3)
    c3 = covariance(df3)
    p1,p2,p3 = prior(train)
    m = [m1,m2,m3]
    c = [c1,c2,c3]
    p = [p1,p2,p3]
    col1 = test[topic[0]].tolist()
    col2 = test[topic[1]].tolist()
    col3 = test[topic[2]].tolist()
    col4 = test[topic[3]].tolist()
    real = test[topic[4]].tolist()
    for i in range(len(col1)):
        x = np.array([[col1[i]],[col2[i]],[col3[i]],[col4[i]]])
        predict.append(cmp1(x,m,c,p,classes))
    confusion(real,predict,classes)
def confusion(real,predict,classes):
    for i in range(len(real)):
        for j in range(len(classes)):
            if real[i] == classes[j]:
                real[i]=j
    for i in range(len(predict)):
        for j in range(len(classes)):
            if predict[i] == classes[j]:
                predict[i] = j
    conf = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(len(real)):
        conf[predict[i]][real[i]] += 1 
    print(conf)
    s1 = 0 
    s2 = 0 
    for i in range(3):
        for j in range(3):
            s1 += conf[i][j]
            if i == j:
                s2 += conf[i][j]
    print('accuracy: ',s2*100/s1,'%')
def normal(x,avg,var):
    return np.exp(-0.5*((x-avg)/var)**2)/(var*(2*np.pi)**0.5)
def naive_bayes(df):
    predict = []
    classes = class_detector(df)
    topic = topic_detector(df)
    test,train = data_divider(df)
    df1,df2,df3 = seprator(train)
    p = prior(train)
    # m1,m2,m3,m4 = mean(df)
    # cov = covariance(df)
    # c1,c2,c3,c4 = cov[0][0],cov[1][1],cov[2][2],cov[3][3]
    col1 = test[topic[0]].tolist()
    col2 = test[topic[1]].tolist()
    col3 = test[topic[2]].tolist()
    col4 = test[topic[3]].tolist()
    real = test[topic[4]].tolist()
    for i in range(len(col1)):
        x1 = col1[i]
        x2 = col2[i]
        x3 = col3[i]
        x4 = col4[i]
        x = [x1,x2,x3,x4]
        m = [mean(df1),mean(df2),mean(df3)]
        v = [variance_(df1),variance_(df2),variance_(df3)]
        predict.append(cmp2(x,m,v,p,classes))
    confusion(real,predict,classes)
def cmp2(x,m,v,p,classes):
    m1,m2,m3 = m
    c1,c2,c3 = v
    p1,p2,p3 = p
    g1 = naive(x,m1,c1,p1)
    g2 = naive(x,m2,c2,p2)
    g3 = naive(x,m3,c3,p3)
    compare = [[0,g1],[1,g2],[2,g3]]
    compare = sorted(compare, key=lambda x: x[1],reverse= True)
    return classes[compare[0][0]]   
def naive(x,avg,var,p):
    return normal(x[0],avg[0],var[0])*normal(x[1],avg[1],var[1])*normal(x[2],avg[2],var[2])*normal(x[3],avg[3],var[3])*p
print('naive bayas:')
naive_bayes(df)
print('-------------------')
print('optimal bayas:')
optimal_bayes(df)