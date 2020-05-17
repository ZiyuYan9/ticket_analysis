from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import csv
import numpy as np
import time 
import sys


def preprocessing(pid, records):
    
    if pid==0:
        next(records)
    
    boro_dict = {'1': 'Manhattan', 
                 '2': 'Bronx', 
                 '3': 'Brooklyn', 
                 '4': 'Queens', 
                 '5': 'Staten Island'}   
    
    reader = csv.reader(records)
    
    for row in reader:
        # Lower street name
        street = row[28].lower()
        # Change boro code into boro name
        boro = boro_dict[row[13]]
        
        for i in [2, 3, 4, 5]:
            if row[i]:
                # Check if the house number is in group like '54-098'
                if row[i].isdigit():
                    # Transfer number like '098' to '98'
                    row[i] = float(row[i])
                    # Group number as integer and house number as decimal 
                       
                else:
                    # Split the group and do the same 
                    first, row[i] = row[i].split('-')
                    row[i] = str(int(row[i]))
                    row[i] = float(first+'.'+row[i])   
            else:
                # If it's null, make it 0.0, SQL could only handle with the same type
                row[i] = 0.0
        
        # yield twice, one for the left and the other for the right
        yield (row[0], street, boro, row[2], row[3], 1)
        yield (row[0], street, boro, row[4], row[5], 0)
        
def dataprocessing(pid, records):
    
    if pid==0:
        next(records)
    
    county_dict = {
        'MAN': 'Manhattan', 'MH':'Manhattan', 'MN': 'Manhattan', 'NEWY': 'Manhattan', 
        'NEW': 'Manhattan', 'Y': 'Manhattan', 'NY': 'Manhattan', 'BRONX': 'Bronx', 
        'BX': 'Bronx', 'BK': 'Brooklyn', 'K': 'Brooklyn', 'KING': 'Brooklyn', 'KINGS': 'Brooklyn',
        'Q': 'Queens', 'QN': 'Queens', 'QNS': 'Queens', 'QU': 'Queens', 'QUEEN': 'Queens',
        'R': 'Staten Island', 'RICHMOND': 'Staten Island'
    } 
    
    reader = csv.reader(records)
    
    for row in reader:
        
        try:
            year = int(str(row[4][-4:]))
        except:
            continue
            
        street = row[24].lower()
        # Transfer county name into boro name
       
        if row[21] and (row[21] in county_dict.keys()):
                boro = county_dict[row[21]]
        else:
            continue
        
        
        if row[23]:
            if row[23].isdigit():
                # Check if the place is on the right or left 
                is_left = int(row[23]) % 2
                house_number = float(row[23])
                
            else:
                # Some times the house number is like 'W', we skip it
                try:
                    first, house_number = row[23].split('-')
                    house_number = str(int(house_number))
                    is_left = int(house_number) % 2
                    house_number = float(first+'.'+house_number)
                        
                except:
                    continue
        else:
            continue
                
        yield (year, street, boro, house_number, is_left)
        
def formatprocessing(records):
    for r in records:
        if r[0][1]==2015:
            yield (r[0][0], (r[1], 0, 0, 0, 0))
        elif r[0][1]==2016:
            yield (r[0][0], (0, r[1], 0, 0, 0))
        elif r[0][1]==2017:
            yield (r[0][0], (0, 0, r[1], 0, 0))
        elif r[0][1]==2018:
            yield (r[0][0], (0, 0, 0, r[1], 0))
        elif r[0][1]==2019:
            yield (r[0][0], (0, 0, 0, 0, r[1]))
        else: 
            yield (r[0][0], (0, 0, 0, 0, 0))

def compute_ols(y, x=list(range(2015,2020))):
    x, y = np.array(x), np.array(y)
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
 
    Y = np.sum(y*x) - n*m_y*m_x 
    X = np.sum(x*x) - n*m_x*m_x
 
    coef = Y / X
    return float(str(coef))


if __name__ == "__main__":
    start_time = time.time()
    output = sys.argv[1]
    
    sc = SparkContext()
    spark = SparkSession(sc)

    centerline = sc.textFile('hdfs:///tmp/bdm/nyc_cscl.csv') \
    .mapPartitionsWithIndex(preprocessing)
    violations = sc.textFile('hdfs:///tmp/bdm/nyc_parking_violation/') \
    .mapPartitionsWithIndex(dataprocessing)
    
    vio = spark.createDataFrame(violations, ('year', 'street', 'boro', 'house_number', 'is_left'))
    cline = spark.createDataFrame(centerline, ('pysicalID', 'street', 'boro', 'low', 'high', 'is_left'))
    condition = [vio.boro == cline.boro, 
             vio.street == cline.street, 
             vio.is_left == cline.is_left, 
             (vio.house_number >= cline.low) & (vio.house_number <= cline.high)]
    df = cline.join(vio, condition, how='left').groupBy([cline.pysicalID, vio.year]).count()
    
    df.rdd.map(lambda x: ((int(x[0]), x[1]), x[2])) \
            .mapPartitions(formatprocessing) \
            .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4])) \
            .sortByKey() \
            .mapValues(lambda y: y + (compute_ols(y=list(y)),)) \
            .map(lambda x: ((x[0],) + x[1])) \
            .map(lambda x: (str(x)[1:-1])) \
            .saveAsTextFile(output)
