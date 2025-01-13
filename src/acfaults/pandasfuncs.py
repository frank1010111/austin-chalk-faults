# Copyright 2011-2018 Frank Male
#This file is part of Fetkovich-Male fit which is released under a proprietary license
#See README.txt for details
import numpy as np
import pandas as pd
import well

def pandify_unclean(field):
    for w in field:
        for key,val in w.__dict__.iteritems():
            if key in ('API','UWI','Entity'):
                w.__dict__[key]=str(val)
            else:
                try:
                    w.__dict__[key]=float(val)
                except:
                    pass
            
    W = [w.__dict__ for w in field]
    return pd.DataFrame(W).set_index('API')

def pandify(field):
    W = [w.__dict__ for w in field]
    return pd.DataFrame(W).set_index('API')

def unpandify(field_pd):
    field = []
    for api,w in field_pd.iterrows():
        start_date = w['start_date']
        if 'production' in w.index:
            production = w['production']
        else:
            production = []
        kwargs = {k:v for k,v in w.iteritems() if k not in ('start_date','production')}
        kwargs['API']=api
        field.append(well.well(start_date,production,**kwargs))
    return field

def Commonstart(df,dropzeros=False):
    if dropzeros:
        data = [np.array(df.iloc[i].replace(0,np.nan).dropna()) for i in range(len(df))]
    else:
        data = [np.array(df.iloc[i].dropna()) for i in range(len(df))]
    index = df.index
    return pd.DataFrame(data,index)
