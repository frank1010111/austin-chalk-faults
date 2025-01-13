# Copyright 2011-2024 Frank Male
"""
Script for gathering data from directional surveys provided by IHS.

Data comes in the form of individual survey point measurements, generally in 50 ft spacing
including every well in the field and for each survey including measured depth from 0 ft to
the total measured depth of the lateral part
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from itertools import chain
import pyproj
from shapely.geometry import LineString

import dask.dataframe as dd

default_proj = pyproj.Proj('epsg:32120')

# Put everything together
def process_headers(wells,survey_points, proj=None, close_dist=1200, surf_dist=50, time_dist_m=1):
    """
    Uses surface, bottomhole locations to calculate first-pass spacing for wells
    
    Inputs
    --------
    wells: pd.DataFrame with surface lateral, longitude columns
    survey_points: DataFrame with IHS style survey points for each well
    proj: Projection for creating X-Y coordinates
    close_dist: distance for wells to be considered close (midpoint)
    surf_dist: distance for wells to be on the same pad
    time_dist_m: max time in months for wells to be classified as simultaneously completed
        times longer than this, one well is a parent, the other a child
        
    Output
    --------
    headers: pd.DataFrame
    """
    
    headers = make_well_location_header(wells, survey_points, proj=proj)

    headers['Pad ID'] = find_surface_pads(headers)
    headers['Spacing category'] = assign_spacings_shbh(
        headers, close_dist=close_dist, surf_dist=surf_dist, time_dist_m=time_dist_m, azi_degrees=30
    )
    headers = (
        headers.reset_index(drop=False)
        .merge(
            headers.groupby(['Pad ID']).count().iloc[:,0]
            .rename('number_of_wells')
            .to_frame()
            .reset_index(drop=False),
            how='left',on='Pad ID')
        .set_index('UWI')
    )
    return headers

def surveys_to_gdf(survey_points: pd.DataFrame, surf_loc: pd.DataFrame, crs:str=None):
    """
    Convert directional surveys into an along-lateral geo-dataframe
    
    Inputs
    --------
    survey_points: pd.DataFrame containing 'Measured Depth', 'Deviation Angle', 'Deviation Azimuth',
        'Deviation N/S', 'N/S', 'Deviation E/W', 'E/W' columns
    surf_loc: pd.DataFrame containing 'surf_x', 'surf_y' columns
    
    Outputs
    --------
    laterals: GeoDataFrame
        contains a geometry with the laterals as Linestrings
    """

    survey_points = survey_points.rename(
        columns={'Measured Depth':'Depth Measured','TV Depth':'Depth True Vertical',
                 'Deviation Angle':'Drift Angle','Deviation Azimuth':'Drift Direction',
                 'Deviation N/S':'North South Coordinate Distance','N/S':'North South Coordinate Direction',
                 'Deviation E/W':'East West Coordinate Distance','E/W':'East West Coordinate Direction'
                }
    )

    # condition the survey points
    survey_points = condition_surveys(survey_points)
    geometry = gpd.GeoSeries(
        survey_points.groupby("UWI")
        .apply(filter_survey_to_lateral)
        .reset_index(drop=True)
        .merge(surf_loc[['surf_x', 'surf_y']], left_on='UWI', right_index=True)
        .assign(
            X=lambda x: x.surf_x + x['East West Coordinate Distance'],
            Y=lambda x: x.surf_y + x['North South Coordinate Distance']   
        )
        [['UWI', 'X', 'Y']]
        .groupby('UWI')
        .apply(make_linestring)
    )
    laterals = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    return laterals

def work_clusters(headers,survey_points,verbose=False):
    cluster_numbers = headers.loc[headers.number_of_wells>=2,'Pad ID'].dropna().unique()
    if verbose:
        print('working',len(cluster_numbers),'clusters today')

    for i,cNum in enumerate(cluster_numbers):
        headers = compute_neighbor_distances_for_cluster(cNum,headers.copy(),survey_points)
        if (i % 100 == 0) and verbose:
            print('finished {} clusters'.format(i))
    return headers

def calculate_distances_clusters(raw_headers,raw_survey_points,verbose=False,proj=None):
    survey_points = raw_survey_points.rename(
        columns={'Measured Depth':'Depth Measured','TV Depth':'Depth True Vertical',
                 'Deviation Angle':'Drift Angle','Deviation Azimuth':'Drift Direction',
                 'Deviation N/S':'North South Coordinate Distance','N/S':'North South Coordinate Direction',
                 'Deviation E/W':'East West Coordinate Distance','E/W':'East West Coordinate Direction'
            }
)
    survey_points = condition_surveys(survey_points)
    # horiz_data = get_everything(survey_points)
    headers = process_headers(raw_headers,survey_points,proj=proj)
    headers = work_clusters(headers,survey_points,verbose)
    return headers

def horizontal_averages(df_in,cols=None,
                        deviation_angle=85):
    """General purpose code for grabbing weighted average of properties that might be set along the
    directional survey"""
    #If we ever get log data along a well lateral, this could be handy
    if cols is None:
        cols = ['Deviation Angle']
    df_out = pd.DataFrame(index=df_in.UWI.unique(),
                    columns=['Entrypoint (MD)','Horizontal length','azimuth']+cols,dtype='float')
    df_out['Entrypoint (MD)']=(df_in[df_in['Drift Angle']>deviation_angle]
                               .groupby('UWI')
                               ['Depth Measured']
                               .min()
                               )
    df_out['Horizontal length']=df_in.groupby('UWI')['Depth Measured'].max()-df_out['Entrypoint (MD)']

    def hz_av(api,w,df,columns):
        "Gets weighted average of columns values of row with label api, values w"
        lateral = w[w['Depth Measured']>=df.loc[api,'Entrypoint (MD)']]
        if not isinstance(columns, (list,tuple)):
            columns = [columns]
        out = []
        weights = w['Depth Measured'].diff(1).loc[lateral.index]
        if weights.sum()==0:
            return (np.nan for c in columns)
        return (np.average(lateral[c],weights=weights) for c in columns)
    
    for api,w in df_in.groupby('UWI'):
        v =  hz_av(api,w,df_out,cols)
        for (i,o) in zip(cols,v):
            df_out.loc[api,i] = o
    return df_out

def condition_surveys(df_in,md_var='Depth Measured'):
    df_in.loc[df_in['North South Coordinate Direction']=='S','North South Coordinate Distance']*=-1
    df_in.loc[df_in['East West Coordinate Direction']=='W','East West Coordinate Distance']*=-1
    return df_in.sort_values(['UWI', md_var], ascending=True)

def get_entrypoint(df_in,drift_angle=85,angle_var='Drift Angle',md_var='Depth Measured'):
    "Get MD of point where survey hits drift_angle and well has turned horizontal"
    assert angle_var in df_in.columns
    assert md_var in df_in.columns
    assert drift_angle<=180
    df = df_in[df_in[angle_var]>drift_angle]
    Entrypoint = (df.groupby('UWI')
                  [md_var]
                  .min()
    )
    return Entrypoint

def filter_survey_to_lateral(well, drift_angle=85):
    entry = well.loc[well['Drift Angle'] > drift_angle, 'Depth Measured'].min()
    return well[well['Depth Measured'] >= entry]

def get_length(df_in,drift_angle=85):
    "get horizontal length from surveys"
    Entrypoint = get_entrypoint(df_in, drift_angle)
    s_out = df_in.groupby('UWI')['Depth Measured'].max() - Entrypoint
    s_out.name='length'
    return s_out

def make_linestring(well):
    if well.shape[0] < 2:
        out = None
    else:
        out = LineString([(x,y) for x,y in zip(well.X, well.Y)])
    return out

def get_azimuth(df_in,drift_angle=85):
    "get azimuth for horizontal leg (in degrees from North)"
    Entrypoint = get_entrypoint(df_in,drift_angle)
    
    def calc_azimuth(w):
        api = w.name
        try:
            in_md =  Entrypoint.loc[api]
        except KeyError:        # happens when there is no entrypoint to horizontal
            return np.nan
        
        w = w.set_index('Depth Measured')
        in_N,in_E = w.loc[in_md,['North South Coordinate Distance','East West Coordinate Distance']]
        end_N,end_E = w[['North South Coordinate Distance','East West Coordinate Distance']].iloc[-1]
        # azimuth_rads = np.arctan2(end_N-in_N,end_E-in_E)
        azimuth_rads = np.arctan2(end_E-in_E,end_N-in_N)
        azimuth_deg = np.degrees(azimuth_rads)
        return azimuth_deg

    azimuth = df_in.groupby('UWI').apply(calc_azimuth)

    return azimuth
    
def get_dip(df_in,drift_angle=85):
    "get dip for horizontal leg (in degrees from horizontal)"
    Entrypoint = get_entrypoint(df_in,drift_angle)
    

    def calc_dip(w):
        api = w.name
        # w = w.set_index('Depth Measured')
        try:
            in_md =  Entrypoint.loc[api]
        except KeyError:        # happens when there is no entrypoint to horizontal
            return np.nan
        w = w[w['Depth Measured'] >= in_md]

        in_TVD = w['Depth True Vertical'].iloc[0]
        out_TVD = w['Depth True Vertical'].iloc[-1]
        in_N,in_E = w[['North South Coordinate Distance','East West Coordinate Distance']].iloc[0]
        end_N,end_E = w[['North South Coordinate Distance','East West Coordinate Distance']].iloc[-1]
        xy_distance = np.sqrt((end_N-in_N)**2+(end_E-in_E)**2)
        dip_rads = np.arctan2(out_TVD-in_TVD,xy_distance)
        dip_degrees = np.degrees(dip_rads)
        return dip_degrees


    dip = df_in.groupby('UWI').apply(calc_dip)

    return dip

def get_TVD(df_in,drift_angle=85):
    "Get TVD for the horizontal of the wells"
    Entrypoint = get_entrypoint(df_in,drift_angle)
    
    def calc_TVD(w):
        api = w.name
        try:
            in_md =  Entrypoint.loc[api]
        except KeyError:        # happens when there is no entrypoint to horizontal
            return np.nan
        w = w[w['Depth Measured']>in_md]
        weights = w['Depth Measured'].diff(1).fillna(0)
        if weights.sum()==0:
            print('no weights')
            return np.nan
        return np.average(w.loc[:, 'Depth True Vertical'], weights=weights)


    TVD = df_in.groupby('UWI').apply(calc_TVD)
    return TVD

def get_midpoint(df_in, surf_loc, drift_angle=85, verbose=False):
    """Get xy location of measured depth midpoint along horizontal
    inputs:
    df_in (dataframe): survey point dataframe
    surf_loc (dataframe): surface locations with first column x, second column y, and index API
    drift_angle (float): angle at which the well reaches horizontal
    verbose (bool): if True, prints problems that cause each NaN

    output:
    midpoint (dataframe): dataframe with midpoints. Columns are ['x_midpoint','y_midpoint']
    """
    Entrypoint = get_entrypoint(df_in, drift_angle)

    def calc_midpoint(w):
        api = w.name
        try:
            in_md = Entrypoint.loc[api]
        except KeyError:
            if verbose:
                print('Cannot find entrypoint MD for well',api)
            return pd.Series( [np.nan,np.nan] )
        w = w.loc[w['Depth Measured']>= in_md]
        mid_md = (in_md + w['Depth Measured'].max())/2
        x = np.interp(mid_md,w['Depth Measured'], w['East West Coordinate Distance'])
        y = np.interp(mid_md,w['Depth Measured'], w['North South Coordinate Distance'])
        try:
            x0,y0 = surf_loc.loc[api]
            return pd.Series( [x0+x, y0+y] )
        except KeyError:
            if verbose:
                print('Cannot find surface location for well',api)
            return pd.Series( [np.nan,np.nan] )

    midpoint = df_in.groupby("UWI").apply(calc_midpoint)
    midpoint.columns = ['x_midpoint','y_midpoint']
    return midpoint

def get_horiz_line(df_in, surf_loc, drift_angle=85):
    "Get xy coordinates of horizontal well path"
    Entrypoint = get_entrypoint(df_in, drift_angle)
    horizontal = pd.Series(index=df_in['UWI'].unique(),name='Horizontal', dtype='O')
    for api,w in df_in.groupby('UWI'):
        #w = w.set_index('Depth Measured')
        try:
            in_md = Entrypoint.loc[api]
        except KeyError:
            continue
        w = w.loc[w['Depth Measured']>= in_md]
        x = w['East West Coordinate Distance'] + surf_loc.loc[api, 'X']
        y = w['North South Coordinate Distance'] + surf_loc.loc[api, 'Y']
        horizontal[api] = LineString(zip(x,y))
    return horizontal
        

def get_everything(df_in,drift_angle=85):
    """Conditions raw IHS directional surveys, then outputs their lateral length, 
    azimuth, and dip as a dataframe"""
    df_out = pd.DataFrame(index=df_in['UWI'].unique())
    
    df_in = condition_surveys(df_in.copy())
    df_out['Depth Measured'] = get_entrypoint(df_in,drift_angle)
    df_out['TVD Horizontal'] = get_TVD(df_in,drift_angle)
    df_out['Length'] = get_length(df_in,drift_angle)
    df_out['Azimuth'] = get_azimuth(df_in,drift_angle)
    df_out['Dip'] = get_dip(df_in,drift_angle)
    return df_out


def make_well_location_header(wells,survey_points,proj=None):
    if not proj:
        proj = pyproj.Proj(proj='utm',zone=15,ellps='WGS84')
    conv_s = lambda x: pd.Series(proj(x['surf_lon'],x['surf_lat']))
    conv_b = lambda x: pd.Series(proj(x['bh_lon'],x['bh_lat']))
    def azimuth(df):
        op = df['bh_x']-df['surf_x']
        adj = df['bh_y']-df['surf_y']
        adj[adj==0]=1e-6
        add = np.zeros(len(op))
        add=np.pi*(adj<0)
        return np.arctan(op/adj)+add
    def fivehundredpoint(x):
        distx = 500*np.sin(x['azimuth'])
        disty = 500*np.cos(x['azimuth'])
        #return pd.Series([x['surf_x']+distx,x['surf_y']+disty])
        return pd.concat([x['surf_x'] + distx, x['surf_y'] + disty], axis=1)
    headers = wells.rename(columns={'Surface Latitude':'surf_lat', 'Surface Longitude':'surf_lon',
                                    'BH Latitude':'bh_lat', 'BH Longitude':'bh_lon',
                                    'Date Completion':'comp_date',
                                    'TVD Horizontal':'tvd_horizontal','Depth Measured':'md',
                                    'TVDSS Horizontal':'tvdss_horizontal',
                                    'Azimuth':'azimuth','Dip':'dip'})
    headers.loc[headers['comp_date'].isna(), 'comp_date'] = headers['start_date']
    try:
        headers = headers.assign(comp_year = lambda x: x.comp_date.dt.year)
    except AttributeError:
        headers['comp_date'] = pd.to_datetime(headers['comp_date'])
        headers = headers.assign(comp_year = lambda x: x.comp_date.dt.year)
    headers = headers[['surf_lat', 'surf_lon', 'bh_lat', 'bh_lon', 'tvd_horizontal',
                       'tvdss_horizontal', 'md', 'azimuth', 'dip',
                       'start_date', 'comp_date', 'comp_year']]
    surf = headers.apply(conv_s,axis=1)*3.280839895
    bh = headers.apply(conv_b,axis=1)*3.280839895
    surf.columns = ['surf_x','surf_y']
    bh.columns = ['bh_x','bh_y']

    # midpoints = (dd.from_pandas(survey_points.set_index(), npartitions=ncores)
    #              get_midpoint(survey_points,surf)
    #              )
    midpoints = get_midpoint(survey_points, surf)
    
    midpoints.columns = ['midpoint_x','midpoint_y']
    headers = (headers.join(surf)
               .join(bh)
               .join(midpoints)
               )

    # headers['midpoint_x'] = headers[['bh_x','surf_x']].mean(axis=1)
    # headers['midpoint_y'] = headers[['bh_y','surf_y']].mean(axis=1)
    # headers['azimuth'] = azimuth(headers)

    #headers['azimuth'] = get_azimuth(survey_points)
    
    #headers[['500_x','500_y']] = fivehundredpoint(headers)

    headers.index.rename('UWI',inplace=True)
    return headers
    
def find_neighbordistances(row,tree,x_key='X',y_key='Y'):
    pos = [row[x_key],row[y_key]]
    query = tree.query(pos,[2])
    #print query[0][0]
    return query[0][0]

def find_surface_pads(headers,max_dist=50):
    tree_surf = cKDTree(headers[['surf_x','surf_y']])
    near_surf = tree_surf.query_pairs(max_dist)
    
    nearsurf_list = set(chain.from_iterable(near_surf))
    #print(len(nearsurf_list),'total wells')
    pads = {i:set((i,j)) for i,j in near_surf}
    for i,j in near_surf:
        for k,l in near_surf:
            if k in (i,j) or l in (i,j):
                pads[i].update((i,j,k,l))

    #print(len(list(chain.from_iterable(pads.values()))),'wells before cleaning')
    pads_clean = pads.copy()
    for i,iv in pads.items():
        for k,kv in pads.copy().items():
            if i==k:
                continue
            if i in kv:
                try:
                    pads_clean[k].update(pads_clean.pop(i,iv))
                except KeyError:
                    for m in pads_clean.keys():
                        if m in kv | iv:
                            pads_clean[m].update(kv|iv)
                            break
    #print(len(list(chain.from_iterable(pads_clean.values()))),'wells after cleaning')
    pad_assign = pd.Series(index=headers.index)
    for p,vals in pads.items():
        for v in vals:
            pad_assign.iloc[v] = p
    return pad_assign

def assign_spacings_shbh(headers,close_dist=500,surf_dist=50,time_dist_m=1,
                         azi_degrees=10):
    tree_surf = cKDTree(headers[['surf_x','surf_y']])
    tree_midpoint = cKDTree(headers[['midpoint_x','midpoint_y']])
    # tree500ft = cKDTree(headers[['500_x','500_y']])
    
    #headers['nearest_distance_surf'] = headers.apply(lambda row:
    #                                                 find_neighbordistances(row,tree_surf,
    #                                                                       'surf_x','surf_y'),
    #                                        axis=1)
    #headers_nearest_distance_midpoint = headers.apply(lambda row:
    #                                                  find_neighbordistances(row,tree_midpoint,
    #                                                                        'midpoint_x','midpoint_y'),
    #                                        axis=1)

    near_surf = tree_surf.query_pairs(surf_dist)
    #near_azi = tree500ft.query_pairs(500 * np.sin(azi_degrees * np.pi / 180))
    near_azi = set()
    for i,j in near_surf:
        ai,aj = headers['azimuth'].iloc[[i,j]]
        try:
            if abs(ai-aj) < azi_degrees:
                near_azi.add((i,j))
        except TypeError:
            #print(e)
            continue

    near_both = near_surf & near_azi
    near_both_list = list(chain.from_iterable(near_both))
    near_both_set = set(near_both_list)
    near_midpoint = tree_midpoint.query_pairs(close_dist)
    
    has_multiple_neighbors = set([i for i in near_both_set if near_both_list.count(i)>1])
    
    clusters = set()
    infill = set()
    close_spaced = set()
    for i,j in near_both:
        di,dj = headers['comp_date'].iloc[[i,j]]
        try:
            if abs(di-dj)<pd.Timedelta(f'{30.5*time_dist_m} days'):
                # if neighbors and completion time close
                if i in has_multiple_neighbors or j in has_multiple_neighbors:
                    #if we have a group of at least 3 close-spaced wells
                    clusters |= set((i,j))
                ##else: #close spaced
            elif abs(di-dj)>pd.Timedelta('182 days'):
                #if neighbors and completion time far
                if di<dj:
                    infill.add(j)
                else:
                    infill.add(i)
        except TypeError:
            continue
            
    infill &= has_multiple_neighbors
    infillcluster = clusters & infill
    infill -= clusters
    clusters -= infillcluster
    close_spaced = (set(chain.from_iterable(near_midpoint | near_azi)) 
                    - (infill | clusters | infillcluster)
                   )
    well_spacing = pd.Series(index=headers.index)
    well_spacing.fillna('single',inplace=True)
    for n,w in {'cluster':clusters,'infill':infill,'close':close_spaced,
                'infill cluster':infillcluster}.items():
        well_spacing.iloc[list(w)] = n
    return well_spacing

def gather_survey_point_locations_for_wells(wellNumber1,wellNumber2,survey_points,clusters):
    
    survey_points = survey_points.rename(columns={'Depth Measured':'Depth','North South Coordinate Distance':'NS',
                                                    'East West Coordinate Distance':'EW'})[['UWI','Depth','NS','EW']]
    survey_points1 = survey_points[survey_points.UWI==wellNumber1].copy()
    survey_points2 = survey_points[survey_points.UWI==wellNumber2].copy()
    survey_points1 = survey_points1[survey_points1['Depth']>=clusters.loc[wellNumber1,'md']]
    survey_points2 = survey_points2[survey_points2['Depth']>=clusters.loc[wellNumber2,'md']]

    # get surface distance between wells
    dx,dy = clusters.loc[wellNumber2,['surf_x','surf_y']]-clusters.loc[wellNumber1,['surf_x','surf_y']]
    
    #adjust second well survey points to put origin at 0,0 for first well
    survey_points2['NS'] += dy
    survey_points2['EW'] += dx
    
    return(survey_points1, survey_points2)
    
def compute_dist_between_laterals(survey_points1,survey_points2):
    
    distances = cdist(survey_points1[['NS','EW']].values,
                      survey_points2[['NS','EW']].values)
    
    return pd.DataFrame(distances)
    
def compute_mean_median_min_distance_between_2_wells(wellNumber1,wellNumber2,survey_points,clusters):
    
    # get survey points
    survey_points1, survey_points2 = gather_survey_point_locations_for_wells(wellNumber1,wellNumber2,
                                                                             survey_points,clusters)
    
    # compute all possible distances between 2 wells
    distances = compute_dist_between_laterals(survey_points1,survey_points2)
    
    # compute the mean median minimum distance between the two wells
    mean_median_min_distance = (distances.min(axis=0).median() + distances.min(axis=1).median())/2.0
    
    return (mean_median_min_distance)

def compute_neighbor_distances_for_cluster(cluster_number,clusters,survey_points):
    cluster = clusters[clusters['Pad ID']==cluster_number].sort_values(['azimuth'])

    distances = pd.DataFrame(index=clusters.index,
                             columns = ['min_dist1', 'right_neighbor', 'min_dist2', 'left_neighbor'],
                             dtype = float
    )
    for wellNumber1,wellNumber2 in zip(cluster.index[:-1],cluster.index[1:]):
        min_dist = compute_mean_median_min_distance_between_2_wells(wellNumber1,wellNumber2,
                                                                    survey_points,clusters)
        distances.loc[wellNumber1,'min_dist1']=min_dist
        distances.loc[wellNumber1,'right_neighbor']=wellNumber2
        distances.loc[wellNumber2,'min_dist2']=min_dist
        distances.loc[wellNumber2,'left_neighbor']=wellNumber1
    return distances

def compute_neighbor_distances_closewells(headers: pd.DataFrame, survey_points: pd.DataFrame, max_dist=500,
                                          nthreads=8):
    tree = cKDTree(headers[['midpoint_x','midpoint_y']])
    near_mid = tree.query_pairs(max_dist)

    distances = pd.DataFrame(index=headers.index,
                             columns = ['min_dist', 'nearest_neighbor'],
                             dtype = float
                             )

    pairs = (pd.DataFrame(list(near_mid),columns=['well1','well2'])
             .apply(lambda x: headers.index[x])
             )

    def calc_dist(pair):
        return compute_mean_median_min_distance_between_2_wells(pair[0], pair[1],
                                                                survey_points,headers)
    min_dist = (dd.from_pandas(pairs, npartitions=nthreads)
                .apply(calc_dist, axis=1, meta=('min_dist','f8'))
                .compute()
    )
    

    for i,(j,k) in enumerate(near_mid):
        wellNumber1,wellNumber2 = headers.index[[j,k]]
        dist =  min_dist[i]
        #= compute_mean_median_min_distance_between_2_wells(wellNumber1,wellNumber2,
        #                                                            survey_points,headers)
        w1dist,w2dist = distances.loc[[wellNumber1,wellNumber2],'min_dist']
        if np.isnan(w1dist) or (dist < w1dist):
            distances.loc[wellNumber1,'min_dist'] = dist
            distances.loc[wellNumber1,'nearest_neighbor'] = wellNumber2

        if np.isnan(w2dist) or (dist < w2dist):
            distances.loc[wellNumber2,'min_dist'] = dist
            distances.loc[wellNumber2,'nearest_neighbor'] = wellNumber1

    return distances
    

def compute_neighbor_vert_distance_clusters(cluster_numbers,clusters,
                                           depthcol='tvd_horizontal',verbose=False):
    clusters = clusters.copy()
    for i,cNum in enumerate(cluster_numbers):
        cluster = clusters[clusters['Pad ID']==cNum].sort_values(['azimuth'])
        for wellNumber1,wellNumber2 in zip(cluster.index[:-1],cluster.index[1:]):
            zdist = clusters.loc[wellNumber1,depthcol] - clusters.loc[wellNumber2,depthcol]
            clusters.loc[wellNumber1,'depth_delta1'] = zdist
            clusters.loc[wellNumber2,'depth_delta2'] = -zdist
        if (i % 100 == 0) and verbose:
            print(f'finished {i} clusters')
    return clusters

def compute_neighbor_vert_distance(headers,neighbor_col, depth_col ='tvd_horizontal'):
    "Get vertical distance between two wells in neighborhood"
    zdist = pd.Series(index=headers.index)
    #zdist.name = 'depth_delta'
    for w in zdist.index:
        n = headers.loc[w, neighbor_col]
        if np.isnan(n):
            # wells that do not have neighbors
            continue
        else:
            zdist.loc[w] = headers.loc[w, depth_col] - headers.loc[n, depth_col]
    return zdist





# Timing questions
def classify_parent_child_neighbor(headers, neighbor_col, date_col, tol=None):
    "Classify whether well is a parent or a child to its nearest neighbor"
    parent = pd.Series(index=headers.index,dtype='O')
    if tol is None:
        tol = pd.Timedelta('45d')
    
    for w in parent.index:
        n = headers.loc[w, neighbor_col]
        if np.isnan(n):
            # wells that do not have neighbors
            continue
        else:
            parent.loc[w] = headers.loc[w, date_col] <= headers.loc[n, date_col] + tol
    return parent.replace({True:'Parent',False:'Child'}).fillna('Single')

def classify_parent_child_withindist(headers, neighbor_tree, date_col, tol=None, distance_upper_bound=1000):
    "Classify whether well is a parent or a child to its nearest neighbor"
    parent = pd.Series(index=headers.index,dtype='O')
    if tol is None:
        tol = pd.Timedelta('45d')
    
    for w in parent.index:
        xy = headers.loc[w,['midpoint_x','midpoint_y']]
        neighbors = neighbor_tree.query_ball_point(xy, distance_upper_bound)
        
        if not neighbors or len(neighbors) <= 1:
            # wells that do not have neighbors
            # Note that query_ball_point gives the indices randomly and will include the original well
            continue
        else:
            parent.loc[w] = headers.loc[w, date_col] <= min(headers.iloc[neighbors][date_col]) + tol
    return parent.replace({True:'Parent',False:'Child'}).fillna('Single')
