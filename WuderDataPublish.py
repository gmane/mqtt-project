#!/usr/bin/python

import urllib2
import json
#import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import time 
from datetime import datetime

#System Variables
url = 'http://api.wunderground.com/api/3a26c3d47e04ad9c/geolookup/conditions/q/TX/San_Antonio.json'




def loadStation(url):
	f = urllib2.urlopen(url)
	json_string =f.read()
	#parsed_json=json.loads(json_string)
	f.close()
	#return parsed_json
	return json_string

def parseTemp(parsed_json):
	temp = parsed_json['current_observation']['temp_f']
	return temp

def parseTime(parsed_json):
	time = parsed_json['current_observation']['observation_time']
	return time

def main():
	mqttc = mqtt.Client(protocol=mqtt.MQTTv31)
	mqttc.connect("10.3.12.48", 1883, 60)
	mqttc.loop_start()
	lasttime = []
	count = 0
	while 0 < 1:
		if (count == 0):
			rep = loadStation(url)
			#temp = parseTemp(rep)
			#thime = parseTime(rep)
			try:
				mqttc.publish("/WundergroundSA", rep, qos=2)
				mqttc.publish("/WundergroundSA/count", "0", qos=2)
				now = datetime.today()
				mqttc.publish("/WundergroundSA/MQTTUpdateTime", str(now), qos=2)
				count = 0;
				print 'Updated Weather Data'
			except:
				print 'Could not parse data. Will retry in 1 second.'
				count = -1;
		else:
			try:
				mqttc.publish("/WundergroundSA", rep, qos=2)
				mqttc.publish("/WundergroundSA/count", str(count), qos=2)
				mqttc.publish("/WundergroundSA/MQTTUpdateTime", str(now), qos=2)
				print 'Republished Old Data. The count is '+ str(count)+' seconds.'
			except:
				print 'Could not publish old data. Will attempt to get new data in 1 second.'
				count = -1
		count = count +1
		if count == (60*4):
			count = 0
		time.sleep(1)
			
#			if thime != lasttime:
#				#publish.single("/SanAntonioTemp/Wunder", payload=("At "+str(thime)+" the temp was "+str(temp)),qos=2,hostname="10.0.0.3")
#				#publish.single("/SanAntonioTemp/Wunder", payload=("At "+str(thime)+" the temp was "+str(temp)),qos=2,hostname="10.3.12.48", port=1883)
#				#mqttc.publish("/SanAntonioTemp/Wunder", (str(thime)+ ' the temperature was ' + str(temp)), qos=2)
#				
#				lasttime = thime
#				count = 0
#				print "At "+str(thime)+ " the temp was "+str(temp)+ " has been published"  
#			else:
#				count = count + 60*4
#				#publish.single("/SanAntonioTemp/Wunder", payload=("Time last updated was "+str(count/60)+ " minutes ago and the temperature was "+str(temp)),qos=2,hostname="10.0.0.3")
#				#publish.single("/SanAntonioTemp/Wunder", payload=("Time last updated was "+str(count/60)+ " minutes ago and the temperature was "+str(temp)),qos=2,hostname="10.3.12.48", port=1883)
#				#mqttc.publish("/SanAntonioTemp/Wunder", ('Last updated '+str(count/60)+" minutes ago and the temperature was "+str(temp)), qos=2)
#				mqttc.publish("/WundergroundSA", rep, qos=2)
#				mqttc.publish("/WundergroundSA/count", count, qos=2)
#				print "Published new count " +str(count/60)
#		except:
#			print 'Could not parse data. Will retry in 15 seconds.'
#		time.sleep(60*4)


 

main()

