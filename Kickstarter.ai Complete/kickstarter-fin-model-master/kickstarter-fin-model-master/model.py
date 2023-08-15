import pickle
import re
import pandas as pd
import sklearn
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

    
model = pickle.load(open('test.pkl','rb'))

df = pd.read_csv('train_upsampled_shuffled.csv')
df_avg = pd.read_csv('companies_allFeatures.csv')
df_avg = df_avg.drop(columns=['permalink', 'name', 'homepage_url', 'status', 'state_code',
                      'region','city', 'founded_at', 'first_funding_at','last_funding_at'])
df_clean = df_avg.dropna()
X_train = df.iloc[:,1:-1]

today = pd.Timestamp.today().normalize()

def calculate_days(date_string):
    today = datetime.today().date()
    specified_date = datetime.strptime(date_string, '%Y-%m-%d').date()
    days_difference = (today - specified_date).days
    return days_difference

def avg_country_data(country_code):
    
    filtered_df = df_clean[(df_clean['country_code'] == country_code) & (df_clean['label'] == 1)]
    funding_total_average = filtered_df['funding_total_usd'].mean()
    funding_duration_avg = filtered_df['funding_duration'].mean()
    funding_rounds_avg = filtered_df['funding_rounds'].mean()
    
    funding_rounds_avg = round(funding_rounds_avg)
    funding_duration_avg = abs(funding_duration_avg)
    list = [funding_total_average, funding_duration_avg, funding_rounds_avg]
    return list
    
def predict_pipeline(category, total_funding, country_code, total_funding_rounds, first_funding_date, last_funding_date):
    # processed_category = re.sub(r'\W+', ' ', category).lower()

    #MAKE THE PANDAS DATAFRAME BELOW THIS USING THE PARAMETERS FROM ABOVE AND ALSO SEE SPECIFICALLY SO THAT EVEN ORDER MATCHES WITH execute
    #Mantej, parameters are matching with the X train - Same order
    suc = 1
    if(total_funding > 2000000.0):
        return suc
    #hello
    processed_first_funding_date = calculate_days(first_funding_date)
    processed_last_funding_date = calculate_days(last_funding_date)

    processed_duration = processed_last_funding_date - processed_first_funding_date
    # processed_duration = processed_last_funding_date - processed_first_funding_date
    # Create a feature vector based on the input parameters THE X_test_con IS THE FINAL FROM THE COPY PASTING THE CODE
    data = {
    'category_list': [category],
    'funding_total_usd': [total_funding],
    'country_code': [country_code],
    'funding_rounds': [total_funding_rounds],
    "('funding_round_permalink', 'count')": [total_funding_rounds],
    'funding_duration' : [processed_duration],
    'first_funding_at_UTC': [processed_first_funding_date],
    'last_funding_at_UTC': [processed_last_funding_date]
    }

    execute = pd.DataFrame(data)
    # execute['category_list'] = execute['category_list'].astype(object)
    # execute['country_code'] = execute['country_code'].astype(object)
    
    
    X_train_text = X_train.category_list
    X_train_country = X_train.country_code
    X_train_nums = X_train.drop(columns=['category_list','country_code'])

    execute_text = execute.category_list
    execute_country = execute.country_code
    execute_nums = execute.drop(columns=['category_list','country_code'])

    execute['category_list'] =execute['category_list'].astype(str)
    X_train['category_list'] = X_train['category_list'].astype(str)
    vectorizer1 = CountVectorizer(min_df=5)
    vectorizer1.fit(X_train.category_list)
    execute_text = vectorizer1.transform(execute.category_list)
    X_train_text = vectorizer1.transform(X_train.category_list)

    execute['country_code']= execute['country_code'].astype(str)
    X_train['country_code'] = X_train['country_code'].astype(str)
    vectorizer2 = CountVectorizer(min_df=1)
    vectorizer2.fit(X_train.category_list)
    execute_country = vectorizer2.transform(execute.country_code)
    X_train_country = vectorizer2.transform(X_train.country_code)

    
    
    execute_text = execute_text.toarray()
    execute_country = execute_country.toarray()
    scaler = sklearn.preprocessing.StandardScaler()
    # X_train_text = X_train_text.toarray()
    # X_train_country = X_train_country.toarray()
    scaler.fit(X_train_nums)
    execute_nums = scaler.transform(execute_nums)
    # X_train_nums = scaler.transform(X_train_nums)
    # X_train_con = np.hstack([X_train_nums, X_train_country, X_train_text])
    execute_con = np.hstack([execute_nums, execute_country, execute_text])
    prediction = model.predict(execute_con)
    return prediction
