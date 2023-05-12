# encoding=utf8
from ..meta_tuning.bird.bird_execution import *
import os
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import warnings
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
        self.exec_result = []
    
    def result_callback(self, result):
        self.exec_result.append(result)
    
    def execute_sql(self, sql, db_path):
        # Connect to the database
        conn = sqlite3.connect(db_path)
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def execute_model(self, sql, db_place, idx):
        try:
            result = func_timeout(30.0, self.execute_sql, args=(sql, db_place))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [('timeout', )]
        except Exception as e:
            print(f'except:{e}')
            result = [('error', )]

        result = {'sql_idx': idx, 'results': result}
        return result
    
    def run_sql_parallel(self, sql, db_place, num_cpus=1):
        pool = mp.Pool(processes=num_cpus)
        pool.apply_async(self.execute_model, args=(sql, db_place), callback=self.result_callback)
        pool.close()
        pool.join()
    
    def run_sqls_parallel(self, sqls, db_places, num_cpus=1):
        pool = mp.Pool(processes=num_cpus)
        for i, sql in enumerate(sqls):
            # if i == 10:
            #     break
            print(f'*************** processing {i}th sql ***************')
            print(sql)
            pool.apply_async(self.execute_model, args=(sql, db_places[i], i), callback=self.result_callback)
        pool.close()
        pool.join()
    
    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

    def compute_execution_accuracy(self, gt_results, predict_results):
        num_correct = 0
        num_queries = len(gt_results)
        mismatch_idx = []

        for i, result in enumerate(gt_results):
            if set(result['results']) == set(predict_results[i]['results']):
                num_correct += 1
            else:
                mismatch_idx.append(i)

        return (num_correct / num_queries) * 100
    
    def flatten_sqls(self, golds):
        sqls = []
        # db_ids = []
        db_places = []
        for result_items in golds:
            sqls.append(result_items['query'])
            # db_ids.append(result_items['db_id'])
            db_places.append(result_items['db_path'] + '/' + result_items['db_id'] + '/' + result_items['db_id'] + '.sqlite')

        return sqls, db_places


    def evaluate(self, preds, golds, section):
        if self.args.seq2seq.target_with_db_id:
            # Remove database id from all predictions
            preds = [pred.split("|", 1)[-1].strip() for pred in preds]
        gold_sqls, db_places = self.flatten_sqls(golds=golds)
        pred_sqls = preds
        
        # just for debugging:
        pred_sqls[-1] = gold_sqls[-1]
        
        self.run_sqls_parallel(pred_sqls, db_places, num_cpus=120)
        pred_results = self.sort_results(self.exec_result)
        
        self.exec_result = []
        self.run_sqls_parallel(gold_sqls, db_places, num_cpus=120)
        
        gold_results = self.sort_results(self.exec_result)
        exec_accuracy = self.compute_execution_accuracy(gt_results=gold_results, predict_results=pred_results)
        import pdb
        pdb.set_trace()
        return {**exec_accuracy}


