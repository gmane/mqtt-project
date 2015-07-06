#! /usr/bin/env python

import paho.mqtt.client as client
import json
import MySQLdb as db


#Initialized Variables

commandmysql = 'INSERT INTO incoming_data (time, temp, e_load) VALUES (%s, %s, %s)'

def SQL_Connect():
	USER = 'bestlab'
	PASS = 'bestlab'
	HOST = 'localhost'
	DATABASE = 'primarydb'
	conn = db.connect(host = HOST, user = USER, passwd = PASS, db = DATABASE)
	return conn	

def on_connect(mqttc, userdata, rc):
	#This is the callback that informs the actions that the mqtt client takes when connecting to the broker
	#In this case it subscribes to the topics as below.
	print("Connected with the result code " +str(rc))
	mqttc.subscribe("/LOADFORECAST/#",qos=0)


def on_message(mqttc, userdata, msg):
	#This is the callback of what happens when a message arrives to the subscriber
	try:
		#The message is printed if it can be parsed, then stored into the MQTT client
		#print(msg.topic+ " "+ str(msg.payload))
		if (msg.topic == "/LOADFORECAST/TIME"):
			mqttc.tyme = msg.payload
		elif(msg.topic == "/LOADFORECAST/TEMP"):
			mqttc.temp = msg.payload
		elif(msg.topic == "/LOADFORECAST/LOAD"):
			mqttc.load = msg.payload
		else:
			#If this error is displayed, then either the subscriptions aren't set up correctly or something is really wrong
			print "Topic Error. How did this happen?"
		
		#These tests establish that both the update time, and the information in the json have been updated	
		if mqttc.tyme != []:
			if mqttc.temp != []:
				if mqttc.load != []:
					try:
						mqttc.sqlconn.cursor().execute(commandmysql, (mqttc.tyme,mqttc.temp,mqttc.load))
						mqttc.sqlconn.commit()
						print 'Data stored into database'
					except:
						print 'SQL Storage Error'
					
					mqttc.load = []
					mqttc.temp = []
					mqttc.tyme = []
	
	except:
		print "Packet Read Error"
		

def on_log(mqttc, obj, level, string):
	print(string)

def on_disconnect(mqttc, obj, rc):
	print rc
	print obj
	mqttc.reconnect()

	
def main():
	
	#This block initializes the MQTT Client and all of the aspects defined above.
	mqttc = client.Client(protocol=client.MQTTv31)
	mqttc.on_connect = on_connect
	mqttc.on_message = on_message
	#mqttc.on_log = on_log
	mqttc.on_disconnect = on_disconnect
	mqttc.sqldb = SQL_Connect
	mqttc.sqlconn = mqttc.sqldb()
	mqttc.tyme = []
	mqttc.temp = []
	mqttc.load = []
	
	#This block connects to the broker and starts the data subscription
	mqttc.connect("10.3.12.48", 1883, 60)
	mqttc.loop_start()
	while True:
		try: 
			mqttc.loop_forever()
		except:
			print "Timeout or other error"
			mqttc.reconnect()
			
if __name__ == '__main__':
	main()