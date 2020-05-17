#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import csv
import sys
from itertools import chain
import numpy as np
import statsmodels.api as sm
from pyspark.sql.functions import broadcast

# get key by searching the value
def get_key (dict, value):
    return [k for k, v in dict.items() if value in v]

def processTicket(index, record):
    
    # skip the first row
    if index==0:
        next(record)
    
    reader = csv.reader(record)
    
    # create a dictionary of Violation County
    county_dic = {'MANHATTAN': ['MAN','MH','MN','NEWY','NEW Y','NY'],
                 'BRONX': ['BRONX','BX'],
                 'BROOKLYN': ['BK','K','KING','KINGS'],
                 'QUEENS': ['Q','QN','QNS','QU','QUEEN'],
                 'STATEN ISLAND': ['R','RICHMOND']}
    
    for row in reader:
        
        if row[21] and row[23] and row[24]:
            # extract year
            try:
                year = int(str(row[4][-4:]))
            except:
                continue
                
            # get the county info
            if row[21] in list(chain(*county_dic.values())):
                county = get_key(county_dic, row[21])[0]
            
            # process the house number
            try:
                # select normal house number
                real_house_number = float(row[23])
                if_odd_left = real_house_number%2
            except:
                try:
                # in this case, the value will be a tuple
                    line = row[23].split('-')[0]
                    number = row[23].split('-')[1]
                    if_odd_left = int(number)%2
                    # transfer it to a float value for better matching the range in cscl dataset
                    real_house_number = float(line+'.'+number)
                except:
                    # invalid value
                    continue
                    
            # get the street info
            street = row[24].lower()
            
            yield (year, county, street, real_house_number, if_odd_left)
            
def processCSCL(index, record):
    
    # skip the first row
    if index==0:
        next(record)
        
    reader = csv.reader(record)
    
    # create a dictionary of the County
    county_dic = {'1': 'MANHATTAN',
                 '2': 'BRONX',
                 '3': 'BROOKLYN',
                 '4': 'QUEENS',
                 '5': 'STATEN ISLAND'}
    
    for row in reader:

        if row[0] and row[2] and row[3] and row[4] and row[5] and row[10] and row[13] and row[28]:
            # get physical ID
            ID = int(row[0])
            
            # get the borough/county data
            county = county_dic[row[13]]
            
            # get the street information, both full_street and st_label for precise match
            street = [row[10].lower(), row[28].lower()]
            
            # process the range columns
            for i in range(2,6):
                # first process those data which look like 177-056
                if '-' in row[i]:
                    line = row[i].split('-')[0]
                    number = row[i].split('-')[1]
                    # transfer it to a float value for better matching the range in cscl dataset
                    row[i] = float(line+'.'+number)
                else:
                    # if the value is a number
                    row[i] = float(row[i])
            
            # output both left side data and right side data
            yield (ID, county, street, row[2], row[3], 1.0)
            yield (ID, county, street, row[4], row[5], 0.0)
            
def extractID(index, record):
    if index==0:
        next(record)
        
    reader = csv.reader(record)
    
    for row in reader:
        # get a full list of the physical ID
       yield (int(row[0]), 'id')
        
def addZero(index, record):
    for row in record:
        if row[1] == 2019:
            yield (row[0], (0,0,0,0,row[2]))
        elif row[1] == 2018:
            yield (row[0], (0,0,0,row[2],0))
        elif row[1] == 2017:
            yield (row[0], (0,0,row[2],0,0))
        elif row[1] == 2016:
            yield (row[0], (0,row[2],0,0,0))
        elif row[1] == 2015:
            yield (row[0], (row[2],0,0,0,0))
        else:
            yield (row[0],(0,0,0,0,0))
            
def calculateCOEF(y):
    x= np.array(range(2015, 2020))
    Y = np.array(y)
    coef = sm.OLS(Y, x).fit().params[0]
    return coef

if __name__ == "__main__":
    
    output = sys.argv[1]
    
    sc = SparkContext()
    spark = SparkSession(sc)
    
    ticket = 'hdfs:///tmp/bdm/nyc_parking_violation/'
    ticket_info = sc.textFile(ticket, use_unicode=True)
    
    cscl = 'hdfs:///tmp/bdm/nyc_cscl.csv'
    cscl_info = sc.textFile(cscl, use_unicode=True)
    
    ticket_data = ticket_info.mapPartitionsWithIndex(processTicket)
    cscl_data = cscl_info.mapPartitionsWithIndex(processCSCL)
    
    physical_id = cscl_info.mapPartitionsWithIndex(extractID)
    
    ticket_format = spark.createDataFrame(ticket_data, ('year', 'county', 'street', 'real_house_number', 'if_odd_left'))
    cscl_format = spark.createDataFrame(cscl_data, ('ID', 'county', 'street', 'low', 'high', 'if_odd_left'))
    id_format = spark.createDataFrame(physical_id, ('Physical_ID', 'nothing')).drop('nothing')
    
    conditions = [ticket_format.county == cscl_format.county,
             (ticket_format.street == cscl_format.street[0]) | (ticket_format.street == cscl_format.street[1]), 
             (ticket_format.real_house_number >= cscl_format.low) & (ticket_format.real_house_number <= cscl_format.high),
             ticket_format.if_odd_left == cscl_format.if_odd_left]
    
    total_data = ticket_format.join(broadcast(cscl_format), conditions, how='inner')
    
    # get the size of matched tickets
    print(len(total_data.collect()))
    
    data_with_value = total_data.groupBy([cscl_format.ID, ticket_format.year]).count()
    data_with_full_ID = id_format.join(data_with_value, id_format.Physical_ID == data_with_value.ID, how='left').drop('ID')
    
    data_with_full_ID.rdd.map(lambda x: (x[0], x[1], x[2])).mapPartitionsWithIndex(addZero)         .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4]))         .sortByKey()         .mapValues(lambda y: list(y)+[calculateCOEF(y)])         .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5]))         .saveAsTextFile(output)






