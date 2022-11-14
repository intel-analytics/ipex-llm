#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import pandas as pd
import click

@click.command()
@click.option('--dataset', default='/ppml/work/data/dataset.csv')
@click.option('--env', default='native')
def main(dataset, env):

    df = pd.read_csv(dataset)
    join_df = pd.read_csv("/ppml/examples/pandas/join_data.csv")
    pd.merge(df, join_df, how = 'left', left_on = 'Inrichting', right_on = 'type')

    df = pd.read_csv(dataset)

    cond =  (df['Voertuigsoort'].str.contains('Personenauto')) | ((df['Merk'] == 'MERCEDES-BENZ') | (df['Merk'] == 'BMW')) | (df['Inrichting'].str.contains('hatchback')) | (df['Handelsbenaming'] == 'FIDDLE') | ((df['Eerste kleur'] == 'GRIJS') | (df['Eerste kleur'] == 'BEIGE'))


    times = 5

    #Complex select statement with UDFs
    print("Complex select")
    start = time.time()

    for i in range(times):
        df_copy = df
        df_copy[cond]

    end = time.time()
    print("Time elapsed: ", round(end - start,4), "s")
    time1 = round(end - start,3)/times

    #Sorting the dataset
    print("Sorting the dataset")
    start = time.time()

    for i in range(times):
        df_copy = df
        df_copy.sort_values(by = "Datum tenaamstelling")

    end = time.time()
    print("Time elapsed: ", round(end - start,4), "s")
    time2 = round(end - start,3)/times

    #Joining the dataset
    print("Joining the dataset")
    join_df = pd.read_csv("/ppml/examples/pandas/join_data.csv")

    start = time.time()

    for i in range(times):
        df_copy = df
        pd.merge(df_copy, join_df, how = 'left', left_on = 'Inrichting', right_on = 'type')

    end = time.time()
    print("Time elapsed: ", round(end - start,4), "s")
    time3 = round(end - start,3)/times

    #Self join
    print("Self join")
    start = time.time()

    for i in range(times):
        df_copy = df
        df_copy['Kenteken'].astype(object)
        pd.merge(df_copy, df_copy, how = 'left', left_on = 'Kenteken', right_on = 'Kenteken')

    end = time.time()
    print("Time elapsed: ", round(end - start,4), "s")
    time4 = round(end - start,3)/times

    #Grouping the data
    print("Grouping the data")
    start = time.time()

    for i in range(times):
        df_copy = df
        df_copy = df_copy.groupby(['Voertuigsoort', 'Merk', 'Inrichting', 'Handelsbenaming', 'Eerste kleur']).size()

    end = time.time()
    print("Time elapsed: ", round(end - start,4), "s")
    time5 = round(end - start,3)/times

    f = open("/ppml/tests/pandas/benchmark_" + env + "_dataset", "w")
    f.write(str(time1) + '\n')
    f.write(str(time2) + '\n')
    f.write(str(time3) + '\n')
    f.write(str(time4) + '\n')
    f.write(str(time5) + '\n')

if __name__ == '__main__':

    main()

