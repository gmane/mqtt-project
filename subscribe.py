#! /usr/bin/env python

import paho.mqtt.client as client

def on_connect(mqttc, userdata, rc):
	print("Connected with the result code " +str(rc))
	mqttc.subscribe("/SanAntonioTemp",qos=1)

def on_message(mqttc, userdata, msg):
	try:
		 print(msg.topic+ " "+ str(msg.payload))
	except:
		print "Packet Read Error"
def on_log(mqttc, obj, level, string):
	print(string)

def on_disconnect(mqttc, obj, rc):
	print rc
	print obj
	mqttc.reconnect()

def main():
	mqttc = client.Client(protocol=client.MQTTv31)
	mqttc.on_connect = on_connect
	mqttc.on_message = on_message
	mqttc.on_log = on_log
	mqttc.on_disconnect = on_disconnect

	#mqttc.loop_start()
	mqttc.connect("192.168.2.63", 1883, 60)
	mqttc.loop_start()
	while True:
		try: 
			mqttc.loop_forever()
		except:
			print "Timeout or other error"
			mqttc.reconnect()

if __name__ == '__main__':
	main()

