# Copyright 2011-2018 Frank Male
# This file is part of Fetkovich-Male fit which is released under a
# proprietary license
# See README.txt for details
# system-wide imports
import os
import argparse
import pandas as pd
import numpy as np

# homemade imports - not needed except for legacy compliance
import pandasfuncs


def readfile(fname, sheetname="Monthly Production"):
    if ".xls" in fname:
        RawF = pd.ExcelFile(fname)
        try:
            Raw = pd.read_excel(
                RawF, sheet_name=sheetname, parse_dates=[["Month", "Year"]]
            ).rename(columns={"Month_Year": "Date"})
        except NotImplementedError:
            print("still not implemented parse_dates for excel")
            Raw = pd.read_excel(RawF, sheetname, converters={"Year": str}).assign(
                Date=lambda x: pd.to_datetime(x.Month + x.Year, format="%b%Y")
            )
    else:
        Raw = pd.read_csv(fname, parse_dates=[["Month", "Year"]]).rename(
            columns={"Month_Year": "Date"}
        )
    Raw["Date"] = Raw["Date"] + pd.offsets.MonthEnd(0)
    return Raw


def resample_pennsylvania(prod, switch="2014-12-31"):
    """For Pennsylvania production reporting, at the end of 2014 they switched from
    semiannual reporting to monthly reporting. Unfortunately, the output files include
    fake months where they've split the 6 months' production evenly into each of the 6 months.
    This takes care of that problem.

    Inputs:
    prod: production series
    switch: date that Pennsylvania switched from semiannual to monthly production (should be '2014-12-31')

    Output:
    production where 6 month periods have been summed until the switch date; after the switch date is untouched
    """
    old, new = prod[:switch], prod[switch:][1:]
    if len(old) < 1:
        return prod
    # fix odd counting of quarters
    # (see https://stackoverflow.com/questions/25383397/python-re-sample-at-a-uniform-semiannual-period-equivaent-of-bq-in-pandas-res)
    st = prod.index[0].year
    old[pd.to_datetime("{}-12-1".format(st - 1))] = np.nan
    old = old.resample("2BQ", convention="e").sum().dropna()
    old[prod.index[0]] = np.nan  # hold onto start date of production for fitting
    return old.append(new).sort_index()


def pivot_to_field(Wells, prod_attrs=None, meta_attrs=None, well_id="API"):
    """Convert unpivoted production workbook into wells

    Inputs:
    Wells: spreadsheet with monthly production data and some well header info
    prod_attrs: Names for columns containing production coming from the spreadsheet.
                Defaults to standard IHS outputs.
    meta_attrs: Names for columns containing relevant production header data.

    Output:
    field: as a dataframe containing monthly production data in pandas Series objects and well header data
    """
    if not prod_attrs:
        prod_attrs = ["Gas", "Liquid", "Water"]
    if not meta_attrs:
        meta_attrs = []  # ['Primary Product','Entity','Operator Name']

    Pivoted = Wells.pivot(index=well_id, columns="Date")

    def get_prop(w, propname):
        try:
            return w[propname].dropna().iloc[-1]
        except IndexError:
            return np.nan

    def get_startdate(prod_dict):
        dates = []
        for prod in prod_dict.values():
            if len(prod) > 0:
                dates.append(prod.index[0])
        return min(dates)

    field = []

    # This part is a little slow. Consider functional-izing
    for api, w in Pivoted.iterrows():
        prod_dict = {fluid: w[fluid].dropna().astype("float") for fluid in prod_attrs}
        meta_dict = {meta: get_prop(w, meta) for meta in meta_attrs}
        meta_dict["API"] = api
        meta_dict["start_date"] = get_startdate(prod_dict)

        meta_dict.update(prod_dict)
        field.append(meta_dict)
    return pd.DataFrame(field).set_index("API")


def build_field(infile, usewelldotpy=True):
    """Takes in production files from IHS, spits out a field for analysis

    Inputs:
    infile: path to excel or csv spreadsheet with production data
    usewelldotpy: Stores data in old format using Well class, kept just for legacy

    Output:
    field: pandas DataFrame with wells indexed by unique well identifier, containing production data and
           some well header data that is present in these spreadsheets
    """
    if not isinstance(infile, pd.DataFrame):
        if not isinstance(infile, (list, tuple)):
            infile = [infile]
        Wells = pd.concat([readfile(f) for f in infile])
    else:
        Wells = infile

    Wells = (
        Wells.groupby(["API", "Date"])
        .agg(
            {
                "Gas": "sum",
                "Liquid": "sum",
                "Water": "sum",
                "Ratio Gas Oil": "first",
                "Percent Water": "first",
                "Primary Product": "first",
                "Entity": "first",
            }
        )
        .reset_index()
    )

    field = pivot_to_field(
        Wells,
        prod_attrs=["Gas", "Liquid", "Water"],
        meta_attrs=["Primary Product", "Entity"],
    )
    field = field.rename(columns={"Gas": "gas", "Liquid": "liquids", "Water": "water"})

    if usewelldotpy:
        return pandasfuncs.unpandify(field)
    else:
        return field

    # print(len(Wells),'reports after dropping duplicates')

    # Pivoted = Wells.pivot('API','Date')

    # def get_prop(w,propname):
    #     return w[propname].dropna()[-1]

    # field = []

    # for api,w in Pivoted.iterrows():
    #     gas = w.Gas.dropna().astype('float')
    #     liquids = w.Liquid.dropna().astype('float')
    #     water = w.Water.dropna().astype('float')
    #     try:
    #         if len(gas)>0:
    #             if len(liquids)>0:
    #                 start_date = min(gas.index[0],liquids.index[0])
    #             else:
    #                 start_date = gas.index[0]
    #         else:
    #             start_date = liquids.index[0]
    #     except IndexError as e:
    #         print(api,'Has no oil nor gas production',e)
    #         start_date = water.index[0]
    #     if usewelldotpy:
    #         field.append(well.well(start_date,[],liquids=liquids,gas=gas,
    #                                water=water,
    #                                primary_product=get_prop(w,'Primary Product'),
    #                                Entity=get_prop(w,'Entity'),
    #                                Operator=get_prop(w,'Operator Name'),
    #                                API=api))

    #     else:
    #         field.append({"start_date":start_date,"liquids":liquids,"gas":gas,"water":water,
    #                       "primary_product":get_prop(w,'Primary Product'),
    #                                "Entity":get_prop(w,'Entity'),
    #                                "Operator":get_prop(w,'Operator Name'),
    #                                "API":api})

    # print(len(field),'wells included')
    # if usewelldotpy:
    #     return field
    # else:
    #     return pd.DataFrame(field).set_index('API')


def convert_excels_hdf(infiles, outfile):
    if not isinstance(infiles, (list, tuple)):
        infiles = [infiles]
    excel = [pd.ExcelFile(infile) for infile in infiles]
    try:
        os.remove(outfile)
        print("Overwriting file at ", outfile)
    except OSError:
        print("Creating new file at ", outfile)
    with pd.HDFStore(outfile) as store:
        for s in excel[0].sheet_names:
            df = pd.concat([pd.read_excel(f, s) for f in excel])
            df.loc[:, df.dtypes == "O"] = df.loc[:, df.dtypes == "O"].astype(str)
            store.append(s, df)


if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser(
        description="Convert production data from IHS format to serialized Python-readable .pkl format"
    )
    parser.add_argument(
        "infile", metavar="INFILE", nargs="+", help="IHS style input file"
    )
    parser.add_argument("-o", "--outfile", default="", help="pickle style output file")
    args = parser.parse_args()

    infile = args.infile

    outfile = args.outfile or os.path.splitext(infile[0])[0] + ".pkl"
    # if not args.outfile:
    #     outfile=os.path.splitext(infile[0])[0]+'.pkl'
    # else:
    #     outfile = args.outfile

    field = build_field(infile, False)
    field.to_pickle(outfile)

    # field = build_field(infile)
    # well.pickle_wells(field,outfile)
    # Field = pd.DataFrame(w.__dict__ for w in field)
    # Field.to_pickle(os.path.splitext(infile[0])[0]+'-pd.pkl')
