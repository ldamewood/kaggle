# -*- coding: utf-8 -*-

import pandas as pd

htypes = ['no echo', 'moderate rain', 'moderate rain2', 'heavy rain',
    'rain/hail', 'big drops', 'AP', 'Birds', 'unknown', 'no echo2',
    'dry snow', 'wet snow', 'ice crystals', 'graupel', 'graupel2']

def impute(infile, outfile):
    print('Loading data...')
    df = pd.read_csv(infile, index_col=['Id', 'Group', 'Index'])

    print('Removing features without variance:')
    remove = []
    for col in df.columns:
        if col in ['Id','Group','Index','Expected']:
            continue
        print(col)
        if col in htypes:
            df[col].fillna(False, inplace='True')
        else:
            df[col].fillna(df[col].mean(), inplace='True')
        if df[col].std() < 1.e-5:
            remove.append(col)
            print('Removing column {}'.format(col))
            del df[col]

    df.to_csv(outfile, index_label=['Id','Group','Index'], header=True)

def normalize(infile, outfile):
    print('Loading data...')
    df = pd.read_csv(infile, index_col=['Id','Group','Index'])

    print('Removing features without variance:')
    remove = []
    for col in df.columns:
        if col in ['Id','Group','Index','Expected']:
            continue
        print(col)
        if col in htypes:
            df[col].fillna(False, inplace='True')
        else:
            df[col].fillna(df[col].mean(), inplace='True')
        if df[col].std() < 1.e-5:
            remove.append(col)
            print('Removing column {}'.format(col))
            del df[col]
            continue
        if col in htypes:
            continue
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean)/std

    df.to_csv(outfile, index_label=['Id','Group','Index'], header=True)

if __name__ == '__main__':
    impute('data/train_split_rows.csv', 'data/train_with_impute.csv')
    normalize('data/train_split_rows.csv', 'data/train_normalize.csv')
    impute('data/test_split_rows.csv', 'data/test_with_impute.csv')
    normalize('data/test_split_rows.csv', 'data/test_normalize.csv')
