#! /usr/bin/env python

import MySQLdb as db
import csv

def SQL_Connect():
	USER = 'bestlab'
	PASS = 'bestlab'
	HOST = 'localhost'
	DATABASE = 'primarydb'
	conn = db.connect(host = HOST, user = USER, passwd = PASS, db = DATABASE)
	return conn

def get_max_index(conn):
	cursor = conn.cursor()
	cursor.execute('SELECT MAX(data_id) FROM incoming_data')
	max_index = cursor.fetchall()[0][0]
	cursor.close()
	conn.commit()
	return max_index

def convert_results(results, desired_interval):
	a = len(results)
	b = xrange(0, a-1, desired_interval)
	c = len(b)
	c_results = [0] * c
	for i in b:
		sum_l = results[i][3]+results[i+1][3]+results[i+2][3]
		ave_t = (results[i][2]+results[i+1][2]+results[i+2][2])/3
		c_results[i/3] = [sum_l, ave_t]
	return c_results


def write_input_file(conn, max_index):
	cursor = conn.cursor()
	cursor.execute('SELECT * FROM incoming_data WHERE data_id > ' + str(max_index-144))
	results = cursor.fetchall()
	cursor.close()
	conn.commit()
	c_results = convert_results(results, 3)
	with open('data_out.csv', 'wb') as csvfile:
		data_author = csv.writer(csvfile)
		data_author.writerow(['Load','Temperature'])
		for row in c_results:
			data_author.writerow([row[0],row[1]])
	csvfile.close()

def main():
	conn = SQL_Connect()
	max_index = get_max_index(conn)
	write_input_file(conn, max_index)
	conn.close()

main()