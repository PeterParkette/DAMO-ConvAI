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

def result_callback(result):
    exec_result.append(result)

def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

def execute_model(sql, db_place, idx):
    try:
        result = func_timeout(30.0, execute_sql, args=(sql, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [('timeout', )]
    except Exception as e:
        print(f'except:{e}')
        result = [('error', )]

    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'results': result}
    return result

def run_sql_parallel(sql, db_place, num_cpus=1):
    pool = mp.Pool(processes=num_cpus)
    pool.apply_async(execute_model, args=(sql, db_place), callback=result_callback)
    pool.close()
    pool.join()

def run_sqls_parallel(sqls, db_place, num_cpus=1):
    pool = mp.Pool(processes=num_cpus)
    for i, sql in enumerate(sqls):
        # if i == 10:
        #     break
        print(f'*************** processing {i}th sql ***************')
        print(sql)
        pool.apply_async(execute_model, args=(sql, db_place, i), callback=result_callback)
    pool.close()
    pool.join()

def package_sqls(sql_path, db_name, mode='codex'):
    clean_sqls = []
    if mode == 'codex':
        sql_data = json.load(open(sql_path + db_name + '_sql.json', 'r'))
        clean_sqls.extend(sql_str for idx, sql_str in sql_data.items())
    elif mode == 'gt':
        sqls = open(sql_path + db_name + '.sql')
        sql_txt = sqls.readlines()
        sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        clean_sqls.extend(iter(sql_txt))
    return clean_sqls

def export_sqls(sql_path, db_name):
    sql_data = json.load(open(sql_path + db_name + '.json', 'r'))

    return [sql_item['query'] for sql_item in sql_data]

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_execution_accuracy(gt_results, predict_results):
    num_correct = 0
    num_queries = len(gt_results)
    mismatch_idx = []

    for i, result in enumerate(gt_results):
        if set(result['results']) == set(predict_results[i]['results']):
            num_correct += 1
        else:
            mismatch_idx.append(i)

    return (num_correct / num_queries) * 100