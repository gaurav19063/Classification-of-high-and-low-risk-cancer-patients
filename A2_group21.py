######
''' 
Group 21
Gaurav Lodhi 
'''



import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as mcof
import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
# cole for making a file
def makefile(prediction,data_test,clf,file):
	ids=data_test['ID']
	new_dataframe=pd.DataFrame(columns=['ID','Labels'])
	new_dataframe['ID']=ids
	# print(len(new_dataframe),prediction)
	new_dataframe['Labels']=prediction
	# ans=pd.concat([data_test['ID'],pred],axis=1)
	new_dataframe.head()

	new_dataframe.to_csv(clf+file,index=False)
	# cNN 
def CNN(X_train,Y_train,X_test,data_test,file):
	model = Sequential()
	model.add(Dense(3000, input_dim=318, activation='relu'))
	model.add(Dense(40, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mse', optimizer='adam')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, epochs=50,  verbose=0)
	pred=model.predict_classes(X_test)
	#print(pred)
	makefile(pred,data_test,"CNN",file)
	


def main():
	n = len(sys.argv)
	args=sys.argv
	print(n)
	print(args[0],args[1],args[2])
	data_train=pd.read_csv(args[1])
	data_test=pd.read_csv(args[2])
	# print(data_train.info())
	# print(data_test.info())
	# data_train.head()

	# sns.countplot(data_train['Labels'])
	# plt.show()

	X_train=data_train.drop(['ID','Labels'],axis=1)
	Y_train=data_train['Labels']
	Y_train.head()
	X_test=data_test.drop(['ID'],axis=1)


	CNN(X_train,Y_train,X_test,data_test,args[3])

	# Select K best for feature selection
	select=SelectKBest(chi2, k=150)
	select.fit_transform(X_train, Y_train)

	# cols= select.get_support(indices=True)
	



	feature_names=X_train.columns
	mask = select.get_support() #list of booleans
	new_features = [] # The list of your K best features
	for bool, feature in zip(mask, feature_names):
	    if bool:
	        new_features.append(feature)
	X_train1 = pd.DataFrame(X_train, columns=new_features)
	X_test1 = pd.DataFrame(X_test, columns=new_features)
	# print(new_features)

	x_train,x_val,y_train,y_val=train_test_split(X_train1,Y_train,stratify=Y_train,test_size=.2)
	x_train=X_train1
	y_train=Y_train
	# # print(x_train.info(),x_val.info())
	x_t=pd.concat([x_train,x_train],axis=0)
	y_t=pd.concat([y_train,y_train],axis=0)

# mlp classifier
	# print((x_t.isna().any()),(y_t.isna().any()))
	model = MLPClassifier(random_state=0,hidden_layer_sizes=90, max_iter=35).fit(x_t, y_t)

	pred=model.predict(x_val)
	print("mathew coff of mlp:-",mcof(y_val,pred))

	pred=model.predict(X_test1)
	makefile(pred,data_test.copy(),"MLP",args[3])





	# return
# xgboost classifier
	xgb = XGBClassifier(
	    max_depth=200,
	    gamma=2,
	    eta=0.7,
	    reg_alpha=0.4,
	    reg_lambda=0.5
	)
	xgb=XGBClassifier()
	xgb.fit(x_t, y_t)
	# print(x_val.head())
	pred=xgb.predict(x_val)
	print("methew coff for xgboost:-",mcof(y_val,pred))

	pred=xgb.predict(X_test1)
	makefile(pred,data_test.copy(),"XGB",args[3])

# random fores classifier

	rfc=RFC(bootstrap= True,
	 max_depth=2900,
	 max_features= 3,
	 min_samples_leaf= 5,
	 min_samples_split=10,
	 n_estimators=200)
	rfc.fit(x_t,y_t)
	pred=rfc.predict(x_val)

	print("methew coff for Random Forest:-",mcof(y_val,pred))
	pred=rfc.predict(X_test1)

	makefile(pred,data_test,"Random Forest Classifier",args[3])

	return

	from sklearn.model_selection import GridSearchCV
	# Create the parameter grid based on the results of random search 
	param_grid = {
	    'bootstrap': [True,False],
	    'max_depth': [80,  100, 200,400,800,1100],
	    'max_features': [2, 3,5,10],
	    'min_samples_leaf': [3, 4, 5],
	    'min_samples_split': [8, 10, 12],
	    'n_estimators': [100, 200, 300, 1000]
	}
	# Create a based model
	rf = RFC()
	# Instantiate the grid search model
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

	grid_search.fit(X_train, Y_train)
	grid_search.best_params_
	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid,X_train,Y_train)
	print(grid_accuracy)












main()

