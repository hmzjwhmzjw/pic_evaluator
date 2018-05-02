#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#===============================================================================
#
#         FILE:   app_srd_jwriter_fxhh_sku_image_v2.py
#
#        USAGE: ./app_srd_jwriter_fxhh_sku_image_v2.py
#
#  DESCRIPTION: 智能写作发现好货商品图片表

#      AUTHOR: zhidong.she
#      COMPANY: jd.com
#      CREATED: 20170329
#      Modified By:Zhu Junwei 2018-01-31
#      VERSION: 2.0
#===============================================================================
import sys
import os
import datetime
import time
from pyspark import SparkContext
from pyspark.sql import HiveContext, SparkSession
import concurrent.futures as futures
import pic_filter_v2


dt = os.getenv("BUFFALO_ENV_BCYCLE")
dt = sys.argv[1]

class SkuImage:
    sku_id = 0
    main_image_path = 'defaultmain'
    sub_image_path1 = 'defaultsub1'
    sub_image_path2 = 'defaultsub2'
    score = 60
    model_id = 1
    created = 'today'

    def toString(self):
        #return '''{"sku_id":{0},"main_image_path":"{1}","sub_image_path1":"{2}","sub_image_path2":"{3}","score":{4},"model_id":{5},"created":"{6}"}'''.format(self.sku_id, self.main_image_path, self.sub_image_path1, self.sub_image_path2, self.score, self.model_id, self.created)
        return '{"sku_id":' + str(self.sku_id) + ',"main_image_path":"' + self.main_image_path + '","sub_image_path1":"' + self.sub_image_path1 + '","sub_image_path2":"' + self.sub_image_path2 + '","score":' + str(self.score) + ',"model_id":' + str(self.model_id) + ',"created":"' + self.created + '"}'


def retrieveIntermediateImages(dt, shardNum, currentShard):
#    print ('dt=%s' % dt)
#    sql = '''select sku_id, image_path_options from app.app_srd_jwriter_intermediate_sku_image_da
#        where dt='{0}' and size(image_path_options) > 2 limit 10'''.format(dt)
    sql = '''select a.* from 
                (select sku_id, image_path_options from tmp.zhujunwei_srd_jwriter_org_sku_image_da 
                where dt='{0}' and size(image_path_options) > 2 and pmod(sku_id, {1}) = {2}) a
            left outer join 
                (select sku_id from tmp.zhujunwei_test_srd_jwriter_fxhh_sku_image_di) b 
            on a.sku_id = b.sku_id where b.sku_id is null
        '''.format(dt, shardNum, currentShard)
    print(sql)
    intermediateImages=spark.sql(sql).collect()
    return intermediateImages


def storeFxhhImagesDi(fxhhImages, dt, currentShard):
    df = spark.createDataFrame(fxhhImages)
    df.createOrReplaceTempView("zhujunwei_test_tmp_app_srd_jwriter_fxhh_sku_image_di_{0}".format(currentShard))
    sql = '''insert into table tmp.zhujunwei_test_srd_jwriter_fxhh_sku_image_di partition (dt='{0}') 
        select sku_id, main_image_path, sub_image_path1, sub_image_path2, score, model_id, created from zhujunwei_test_tmp_app_srd_jwriter_fxhh_sku_image_di_{1}'''.format(dt, currentShard)
    print(sql)
    spark.sql("SET hive.exec.dynamic.partition=TRUE")
    spark.sql("SET hive.exec.dynamic.partition.mode=nonstrict")
    spark.sql("SET mapreduce.input.fileinputformat.split.maxsize=268435456")
    spark.sql("SET hive.default.fileformat=Orc")
    spark.sql("SET mapred.max.split.size=1073741824")
    spark.sql("SET hive.exec.reducers.bytes.per.reducer=1073741824")
    spark.sql("SET hive.exec.orc.default.block.size=536870912")
    spark.sql(sql)

def aggregateFxhhImageDa(dt):
    sql = '''insert overwrite table tmp.zhujunwei_test_srd_jwriter_fxhh_sku_image_da partition (dt='{0}') 
            select sku_id, main_image_path, sub_image_path1, sub_image_path2, score, model_id, created from tmp.zhujunwei_test_srd_jwriter_fxhh_sku_image_di where score > 50'''.format(
        dt)
    print(sql)
    spark.sql("SET hive.exec.dynamic.partition=TRUE")
    spark.sql("SET hive.exec.dynamic.partition.mode=nonstrict")
    spark.sql("SET mapreduce.input.fileinputformat.split.maxsize=268435456")
    spark.sql("SET hive.default.fileformat=Orc")
    spark.sql("SET mapred.max.split.size=1073741824")
    spark.sql("SET hive.exec.reducers.bytes.per.reducer=1073741824")
    spark.sql("SET hive.exec.orc.default.block.size=536870912")
    spark.sql(sql)

def processFxhhImages(shardNum, currentShard):
    fxhhImages = []
    intermediateImages = retrieveIntermediateImages(dt, shardNum, currentShard)
    imgLen = len(intermediateImages)
    current = 1

    pic_filter = pic_filter_v2.GetValidPics()
    for interImg in intermediateImages:
        sku = interImg.sku_id
        imgOptions = interImg.image_path_options
        print("T[{0}] Row[{1}/{2}] Processing sku {3} ...".format(currentShard, current, imgLen, sku))
        start = int(round(time.time() * 1000))
        try:
            validImgs = pic_filter(str(sku), imgOptions)
        except:
            print("exception occurs, skip skuid = {0}".format(sku))
            current += 1
            continue
        end = int(round(time.time() * 1000))
        print("time consumed for call filter_pic: {0} ms".format(end - start))
        if len(validImgs) == 3:
            img = SkuImage()
            img.sku_id = sku
            img.main_image_path = validImgs[0]
            img.sub_image_path1 = validImgs[1]
            img.sub_image_path2 = validImgs[2]
            img.score = 80
            img.model_id = 1
            img.created = dt
            print('''valid image is {0}'''.format(img.toString()))
            fxhhImages.append(img)
        elif len(validImgs) == 2:
            img = SkuImage()
            img.sku_id = sku
            img.main_image_path = validImgs[0]
            img.sub_image_path1 = validImgs[1]
            img.sub_image_path2 = 'no_valid_image'
            img.score = 70
            img.model_id = 1
            img.created = dt
            print('''valid image is {0}'''.format(img.toString()))
            fxhhImages.append(img)
        elif len(validImgs) == 1:
            img = SkuImage()
            img.sku_id = sku
            img.main_image_path = validImgs[0]
            img.sub_image_path1 = 'no_valid_image'
            img.sub_image_path2 = 'no_valid_image'
            img.score = 60
            img.model_id = 1
            img.created = dt
            print('''valid image is {0}'''.format(img.toString()))
            fxhhImages.append(img)
        else:
            print("no valid image, set score to zero, skip this sku.")
            img = SkuImage()
            img.sku_id = sku
            img.main_image_path = 'no_valid_image'
            img.sub_image_path1 = 'no_valid_image'
            img.sub_image_path2 = 'no_valid_image'
            img.score = 0
            img.model_id = 1
            img.created = dt
            fxhhImages.append(img)
        current += 1

    print("length of total sku: %s" % imgLen)
    print("length of valid sku: %s" % len(fxhhImages))
    # print("list is: %s" % fxhhImages)
    if len(fxhhImages) > 0:
        storeFxhhImagesDi(fxhhImages, dt, currentShard)

if __name__=='__main__':
    sc=SparkContext(appName="MySpark")

    # 本地模型路径
    hdfs_dirPath = 'hdfs://ns15/user/mart_srd/zhujunwei/'
    tmp_model_path = '/tmp/fxhhmodel/'
    if os.path.exists(tmp_model_path) == False:
        os.makedirs(tmp_model_path)
    data = sc.binaryFiles(hdfs_dirPath)  # Read a directory of binary files from HDFS
    models = data.collect()
    print('model num:',len(models))
    for model in models:
        if model[0].find('pic_evaluator_20180131.pth') > 0:
            with open('/tmp/fxhhmodel/pic_evaluator_20180131.pth','wb') as model1:
                model1.write(model[1])
                print('get pic_evaluator_20180131.pth!')
        if model[0].find('squeezenet1_1-f364aa15.pth') > 0:
            with open('/tmp/fxhhmodel/squeezenet1_1-f364aa15.pth','wb') as model2:
                model2.write(model[1])
                print('get squeezenet1_1-f364aa15.pth!')

    sqlContext = HiveContext(sc)
    spark=SparkSession.builder.appName("HiveDao").enableHiveSupport().getOrCreate()
    #data=get_toutiao('2017-11-22')
    #url=uploadImage("59b8e0fdN8500c4e0.jpg")
    #print(url)
    #dt = '2017-12-03'
    today = datetime.date.today()
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    print ("dt from $BCYCLE, dt={0}".format(dt))
    print ("today={0}".format(today))
    print ("oneday={0}".format(oneday))
    print ("yesterday={0}".format(yesterday))
    if dt is None:
        print ("env BUFFALO_ENV_BCYCLE is null, use yesterday as dt.")
        dt = str(yesterday)
    #dt = '2018-01-04'
    print ("final dt={0}".format(dt))

    process_num = 8
    with futures.ProcessPoolExecutor(max_workers=process_num) as executor:
        future_list = list()
        for i in range(process_num):
            future_list.append(executor.submit(processFxhhImages, process_num, i))

        for future in futures.as_completed(future_list):
            if future.exception():
                print(future.exception())
                
    aggregateFxhhImageDa(dt)








