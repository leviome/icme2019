import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr import SingleFeat
import tensorflow as tf

def model_pool(defaultfilename="c.txt", defaulttarget="label", defaultmodel="AFM"):
    filename=input("filename:")
    if not filename:
        filename = defaultfilename
    data=pd.read_csv(filename)
    #1. smart recogonize
    sparse_features=[]
    dense_features=[]
    for column in data.columns:
        if (data[column].dtype == numpy.float64 or data[column].dtype == numpy.int64):
            dense_features.append(column)
        else:
            sparse_features.append(column)
    target=input("target:")
    if not target:
        target=defaulttarget
    try:
        sparse_features.remove(target)
        dense_features.remove(target)
    except ValueError:
        pass
    print(sparse_features)
    print(dense_features)
    target=[target]
    
    # 0. Remove na values
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.Dense normalize
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # 3.generate input data for model
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    # 4.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    # 5.generate data
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]
    # 6.choose a model
    import pkgutil
    import deepctr.models
    modelnames = [name for _, name, _ in pkgutil.iter_modules(deepctr.models.__path__)]
    modelname = input("choose a model: "+",".join(modelnames)+"\n")
    if not modelname:
        modelname=defaultmodel
    # 7.build a model
    model = getattr(deepctr.models, modelname)({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, final_activation='sigmoid')
    # 8. eval predict
    def auc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy', auc])
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=100, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    
if __name__ == "__main__":
    model_pool()