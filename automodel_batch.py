import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr import SingleFeat
import tensorflow as tf
from keras.callbacks import EarlyStopping

def model_pool(defaultfilename='./input/final_track1_train.txt', defaulttestfile='./input/final_track1_test_no_anwser.txt',
                defaultcolumnname=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'],
                defaulttarget=['finish', 'like'], defaultmodel="AFM", PERCENT=100):
    
    # read file and build sparse feature list
    # fix a bug for version 6 when:
    #     tensorflow.python.framework.errors_impl.InvalidArgumentError: indices[15134,0] = 60990 is not in [0, 59414)
    #      [[{{node sparse_emb_6-music_id/embedding_lookup}} = ..
    #**************Features
    sparse_features=[]
    dense_features=[]
    target=defaulttarget
    
    data = pd.read_csv(defaultfilename, sep='\t', names=defaultcolumnname, iterator=True)
    loop = True
    uniq_dic={}
    while loop:
        try:
            chunk=data.get_chunk(10**7)
            if len(dense_features)+len(sparse_features)==0:
                for column in chunk.columns:
                    if column in defaulttarget:
                        continue
                    if chunk[column].dtype in  [numpy.float_ , numpy.float64]:
                        dense_features.append(column)
                    if chunk[column].dtype in [numpy.int_, numpy.int64]:
                        sparse_features.append(column)
                uniq_dic=dict((feat,[]) for feat in sparse_features)
            
            # get all sparse cases in sample data
            for feat in sparse_features:
                uniq_dic[feat].extend(chunk[feat].unique())
                uniq_dic[feat] = list(set(uniq_dic[feat]))
                
        except StopIteration:
            loop=False
            print('stop iteration for sparse feature engineering')
        
    #6. generate input data for model
    sparse_feature_list = [SingleFeat(feat, len(uniq_dic[feat]))
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    data.close()
    # read file and build sparse feature list
    
    # traing begins
    data = pd.read_csv(defaultfilename, sep='\t', names=defaultcolumnname, iterator=True)
    #1 train file concats
    loop = True
    take=[]
    while loop:
        try:
            chunk=data.get_chunk(10**7)
            if PERCENT < 100 and PERCENT > 0:
                chunk=chunk.take(list(range(min(chunk.shape[0], PERCENT*100))), axis=0)
            
            #***************normal
            #3. Remove na values
            chunk[sparse_features] = chunk[sparse_features].fillna('-1', )
            chunk[dense_features] = chunk[dense_features].fillna(0,)
            #4. Label Encoding for sparse features, and do simple Transformation for dense features
            for feat in sparse_features:
                lbe = LabelEncoder()
                chunk[feat] = lbe.fit_transform(chunk[feat])
            #5. Dense normalize
            if dense_features:
                mms = MinMaxScaler(feature_range=(0, 1))
                chunk[dense_features] = mms.fit_transform(chunk[dense_features])
            #*****************normal
            take.append(chunk)
        except StopIteration:
            loop=False
            print('stop iteration')
    
    train_data=pd.concat(take, copy=False)
    #****************model
    # 6.choose a model
    import pkgutil
    import mdeepctr.models
#     modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.__path__)]
#     modelname = input("choose a model: "+",".join(modelnames)+"\n")
#     if not modelname:
    modelname=defaultmodel
    # 7.build a model
    model = getattr(mdeepctr.models, modelname)({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, final_activation='sigmoid', output_dim=len(defaulttarget))
    # 8. eval predict
    def auc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
    model.compile("adam", loss="binary_crossentropy", metrics=[auc])

        
    train_model_input = [train_data[feat.name].values for feat in sparse_feature_list] + \
                        [train_data[feat.name].values for feat in dense_feature_list]
    train_labels = [train_data[target].values for target in defaulttarget]

    my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-2, patience=2, verbose=1, mode='min')]

    history = model.fit(train_model_input, train_labels,
                batch_size=2**14, epochs=100, verbose=1, callbacks=my_callbacks)

    
    #2 test file       
    test_data = pd.read_csv(defaulttestfile, sep='\t', names=defaultcolumnname, )
    raw_test_data=test_data.copy()
    #data = data.append(test_data)
    test_size=test_data.shape[0]
    print(test_size)
    #***************normal
    #3. Remove na values
    test_data[sparse_features] = test_data[sparse_features].fillna('-1', )
    test_data[dense_features] = test_data[dense_features].fillna(0,)
    #4. Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        test_data[feat] = lbe.fit_transform(test_data[feat])
    #5. Dense normalize
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        test_data[dense_features] = mms.fit_transform(test_data[dense_features])
    #*****************normal
    #test = test_data
    test_model_input = [test_data[feat.name].values for feat in sparse_feature_list] + \
        [test_data[feat.name].values for feat in dense_feature_list]
       
    pred_ans = model.predict(test_model_input, batch_size=2**14)
        
    result = raw_test_data[['uid', 'item_id', 'finish', 'like']].copy()
    result.rename(columns={'finish': 'finish_probability',
                           'like': 'like_probability'}, inplace=True)
    result['finish_probability'] = pred_ans[0]
    result['like_probability'] = pred_ans[1]
    output = "%s-result.csv" % (modelname)
    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
        output, index=None, float_format='%.6f')
    
    return history

if __name__ == "__main__":
    import pkgutil
    import mdeepctr.models
    modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.models.__path__)]
    functions = ["AFM", "DCN", "MLR",  "DeepFM",
           "MLR", "NFM", "DIN", "FNN", "PNN", "WDL", "xDeepFM", "AutoInt", ]
    models_dic = dict((function.lower(),function) for function in functions)
    for modelname in modelnames:
        print(modelname)
        if models_dic[modelname] not in ["PNN"]:
            continue
        history = model_pool(defaultmodel=models_dic[modelname], PERCENT=100)
        print(history.history)
        