#! /usr/bin/env python

import paho.mqtt.client as client

def on_connect(mqttc, userdata, rc):
	print("Connected with the result code "+ str(rc))
	mqttc.subscribe("$SYS/#")

def on_message(mqttc, userdata, msg):
	print(msg.topic+" "+str(msg.payload))

def main():
	mqttc = client.Client()
	mqttc.on_connect = on_connect
	mqttc.on_message = on_message

	mqttc.connect("iot.eclipse.org", 1883, 60)
	mqttc.loop_forever()

if __name__ == '__main__':
	main()
