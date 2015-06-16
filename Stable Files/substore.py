#! /usr/bin/env python

import paho.mqtt.client as client
import json
import MySQLdb as db

def SQL_Connect():
	USER = 'root'
	PASS = 'bestlab'
	HOST = 'localhost'
	DATABASE = 'Wunderground Weather'
	conn = db.connect(host = HOST, user = USER, passwd = PASS, db = DATABASE)
	return conn

def on_connect(mqttc, userdata, rc):
	#This is the callback that informs the actions that the mqtt client takes when connecting to the broker
	#In this case it subscribes to the topics as below.
	print("Connected with the result code " +str(rc))
	mqttc.subscribe("/SanAntonioTemp",qos=0)
	mqttc.subscribe("/SanAntonioTemp/Time",qos=0)
	
	
def on_message(mqttc, userdata, msg):
	#This is the callback of what happens when a message arrives to the subscriber
	add_data = ("INSERT INTO Pymetar_Data (idPymetar_Data, Temp) VALUES (%s, %s)")
	try:
		#The message is printed if it can be parsed, then stored into the MQTT client
		#print(msg.topic+ " "+ str(msg.payload))
		if (msg.topic == "/SanAntonioTemp"):
			print(msg.topic)
			mqttc.weatherjsonold = mqttc.weatherjsonnew
			mqttc.weatherjsonnew = msg.payload
		elif(msg.topic == "/SanAntonioTemp/Time"):
			print(msg.topic+ " "+ str(msg.payload))
			mqttc.weathertimeold = mqttc.weathertimenew
			mqttc.weathertimenew = msg.payload
			try:
				storedb = mqttc.sqlconn.cursor()
				storedb.execute(add_data, (str(mqttc.weathertimenew), str(mqttc.weatherjsonnew)))
				mqttc.sqlconn.commit()
				print "Data stored into database"
			except:
				print "SQL storage error"
		else:
			#If this error is displayed, then either the subscriptions aren't set up correctly or something is really wrong
			print "Topic Error. How did this happen?"
		
		

		
		
	except:
		print "Packet Read Error"
		
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


if __name__ == '__main__':
	main()