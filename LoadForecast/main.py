#! /usr/bin/env python

#These are the imported libraries
import paho.mqtt.client as client
import json
import MySQLdb as db
import create_input as c_i

#Variables related to this
add_data = ("INSERT INTO incoming_data (time, temp, e_load) VALUES (%s, %s, %s)")


def SQL_Connect():
	#This creates a connection to an SQL database
	USER = 'bestlab'
	PASS = 'bestlab'
	HOST = 'localhost'
	DATABASE = 'localstore'
	conn = db.connect(host = HOST, user = USER, passwd = PASS, db = DATABASE)
	return conn

def on_connect(mqttc, userdata, rc):
	#This is the callback that informs the actions that the mqtt client takes when connecting to the broker
	#In this case it subscribes to the topics as below.
	print("Connected with the result code " +str(rc))
	mqttc.subscribe("/SanAntonioTemp",qos=0)
	mqttc.subscribe("/SanAntonioTemp/Time",qos=0)

def on_disconnect(mqttc, obj, rc):
	#Should the mqtt client disconnect, this attempts to reconnect it
	print rc
	print obj
	mqttc.reconnect()

def on_message(mqttc, userdata, msg):
	#This is the callback of what happens when a message arrives to the subscriber
	

def main():

	#This block initializes the MQTT Client and all of the aspects defined above.
	mqttc = client.Client(protocol=client.MQTTv31)
	#Attaches functions
	mqttc.on_connect = on_connect
	mqttc.on_message = on_message
	mqttc.on_disconnect = on_disconnect
	mqttc.sqldb = SQL_Connect
	mqttc.sqlconn = mqttc.sqldb()
	#Attaches variables
	mqttc.weathertimeold = []
	mqttc.weathertimenew = []
	mqttc.weatherjsonold = []
	mqttc.weatherjsonnew = []

	#This block connects to the broker and starts the data subscription
	mqttc.connect("10.3.12.48", 1883, 60)
	mqttc.loop_start()
	while True:
		try: 
			mqttc.loop_forever()
		except:
			print "Timeout or other error"
			mqttc.reconnect()
