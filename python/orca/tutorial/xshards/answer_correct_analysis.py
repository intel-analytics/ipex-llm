import pandas as pd
import numpy as np

# train_df = pd.read_csv(
#     '/kaggle/input/riiid-test-answer-prediction/train.csv',
#     low_memory=False,
#     nrows=10**6,
#     dtype=types
# )
#
# train_df.head()
#
# print('Part of missing values for every column')
# print(train_df.isnull().sum() / len(train_df))
#
# questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
# questions.head()
#
# print('Part of missing values for every column')
# print(questions.isnull().sum() / len(questions))

used_data_types_dict = {
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float16',
    'prior_question_had_explanation': 'boolean'
}

train_df = pd.read_csv(
    '/home/ding/data/answer_correctness/train_debug.csv',
    usecols = used_data_types_dict.keys(),
    dtype=used_data_types_dict,
    index_col = 0
)

features_df = train_df.iloc[:int(9/10 * len(train_df))]
train_df = train_df.iloc[int(9/10 * len(train_df)):]

train_questions_only_df = features_df[features_df['answered_correctly']!=-1]
grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg(
    {
        'answered_correctly': [
            'mean',
            'count',
            'std',
            'median',
            'skew'
        ]
    }
).copy()

user_answers_df.columns = [
    'mean_user_accuracy',
    'questions_answered',
    'std_user_accuracy',
    'median_user_accuracy',
    'skew_user_accuracy'
]

user_answers_df

grouped_by_content_df = train_questions_only_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg(
    {
        'answered_correctly': [
            'mean',
            'count',
            'std',
            'median',
            'skew'
        ]
    }
).copy()

content_answers_df.columns = [
    'mean_accuracy',
    'question_asked',
    'std_accuracy',
    'median_accuracy',
    'skew_accuracy'
]

content_answers_df

features = [
    'mean_user_accuracy',
    'questions_answered',
    'std_user_accuracy',
    'median_user_accuracy',
    'skew_user_accuracy',
    'mean_accuracy',
    'question_asked',
    'std_accuracy',
    'median_accuracy',
    'prior_question_elapsed_time',
    'prior_question_had_explanation',
    'skew_accuracy'
]

target = 'answered_correctly'

train_df = train_df[train_df[target] != -1]

train_df = train_df.merge(user_answers_df, how='left', on='user_id')
train_df = train_df.merge(content_answers_df, how='left', on='content_id')

train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value=False).astype(bool)
train_df = train_df.fillna(value=0.5)

train_df = train_df[features + [target]]
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df.fillna(0.5)

train_df

