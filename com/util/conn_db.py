# -*- coding: UTF-8 -*-

import pymysql
import logging

class ConnMysql:
	def __init__(self, host, port, user, password, db):
		self.host = host
		self.port = port
		self.user = user
		self.password = password
		self.db = db

	def connectMysql(self):
		try:
			self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.password, db=self.db, charset='utf8')
			self.cursor = self.conn.cursor()
			print("connect successful")
		except:
			print("connect mysql error.")

	def queryData(self, sql):
		try:
			self.cursor.execute(sql)
			return self.cursor.fetchall()
		except:
			print("query execute failed")
			return None

	def insertData(self, sql):
		try:
			self.cursor.execute(sql)
		except:
			print("insert execute failed")

	def updateData(self, sql):
		try:
			self.cursor.execute(sql)
			self.conn.commit()
		except:
			self.conn.rollback()

	def closeMysql(self):
		self.cursor.close()
		self.conn.close()