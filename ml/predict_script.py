import pandas as pd
import catboost


def clear_data(df):
    df['CLIENT_REGISTRATION_REGION'] = [x if x in ['Санкт-Петербург', 'Москва',
                                                   'Московская', 'Ленинградская'] else 'Other' for x in df['CLIENT_REGISTRATION_REGION']]
    channel_count = df['POLICY_SALES_CHANNEL'].value_counts()
    df['POLICY_SALES_CHANNEL'] = [x if channel_count[x] >
                                  500 else 'Other' for x in df['POLICY_SALES_CHANNEL']]
    make_count = df['VEHICLE_MAKE'].value_counts()
    df['VEHICLE_MAKE'] = [x if make_count[x] >
                          100 else 'Other' for x in df['VEHICLE_MAKE']]
    model_count = df['VEHICLE_MODEL'].value_counts()
    df['VEHICLE_MODEL'] = [x if model_count[x] >
                           30 else 'Other' for x in df['VEHICLE_MODEL']]
    group_count = df['POLICY_SALES_CHANNEL_GROUP'].value_counts()
    df['POLICY_SALES_CHANNEL_GROUP'] = [x if group_count[x] >
                                        3000 else 'Other' for x in df['POLICY_SALES_CHANNEL_GROUP']]
    df['POLICY_MIN_DRIVING_EXPERIENCE'] = [2021 - x if x >
                                           1900 else x for x in df['POLICY_MIN_DRIVING_EXPERIENCE']]
    df['VEHICLE_ENGINE_POWER'][(df['VEHICLE_ENGINE_POWER'] > 700) | (
        df['VEHICLE_ENGINE_POWER'] <= 0)] = df['VEHICLE_ENGINE_POWER'].mean()
    df['VEHICLE_SUM_INSURED'][df['VEHICLE_SUM_INSURED']
                              <= 0] = df['VEHICLE_SUM_INSURED'].mean()
    df['POLICY_CLM_N'][df['POLICY_CLM_N'] == 'n/d'] = '0'
    df['POLICY_CLM_GLT_N'][df['POLICY_CLM_GLT_N'] == 'n/d'] = '0'
    df['POLICY_YEARS_RENEWED_N'][df['POLICY_YEARS_RENEWED_N'] == 'N'] = 0
    df['POLICY_YEARS_RENEWED_N'] = pd.to_numeric(
        df['POLICY_YEARS_RENEWED_N'], downcast='integer')

    return df


def get_predictions(df):
    pred_df = pd.DataFrame(df['POLICY_ID'])
    bad = ['POLICY_ID', 'POLICY_END_MONTH',
           'POLICY_INTERMEDIARY', 'DATA_TYPE', 'POLICY_IS_RENEWED']  # 'DATA_TYPE',
    X = df.drop(bad, axis=1)
    model = catboost.CatBoostClassifier()
    model.load_model('ctb')

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    pred_df['POLICY_IS_RENEWED'] = y_pred.tolist()
    pred_df['POLICY_IS_RENEWED_PROBABILITY'] = y_pred_proba.tolist()

    return pred_df


df = pd.read_csv('data.txt', sep=';')
train_df = df[df['DATA_TYPE'] == 'TRAIN']
test_df = df[df['DATA_TYPE'] == 'TEST ']
train_pred_df = get_predictions(clear_data(train_df))
test_pred_df = get_predictions(clear_data(test_df))
train_pred_df.to_csv('train_pred_df.csv')
test_pred_df.to_csv('test_pred_df.csv')
