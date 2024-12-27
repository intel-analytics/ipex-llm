import pandas as pd
import ast


# # fruits = ast.literal_eval(fruits)
# set_movie = []
def f(x):
    return ast.literal_eval(x['name'])
df = pd.read_csv("resources/cache.csv")
df['name'] = df.apply(lambda x: f(x), axis=1)

def f(x):
    return ast.literal_eval(x['Genre'])
df['Genre'] = df.apply(lambda x: f(x), axis=1)
# df.apply(lambda x: f(x), axis=1)
filtered_df = df[df['user'] == 1]
genres = ['Comedy']

res_list = []
def f(x):
    for i, n in enumerate(x['Genre']):
        for g in genres:
            if g in n:
                res_list.append(x['name'][i])

filtered_df.apply(lambda x: f(x), axis=1)
print("Hi")