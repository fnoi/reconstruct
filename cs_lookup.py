import pandas as pd

if __name__== '__main__':

    with open('data/beams/aisc-shapes-database-v15.0.csv', 'r') as f:
        # read to pandas, first row is header
        beams = pd.read_csv(f, header=0, sep=';')
        # new dataframe from a selection of columns only
        beams_frame = beams[['tw', 'tf', 'bf', 'd']]
    a = 0