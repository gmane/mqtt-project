#!/usr/bin/python

import urllib2
import json
import paho.mqtt.publish as publish
import time 

#System Variables
url = 'http://api.wunderground.com/api/3a26c3d47e04ad9c/geolookup/conditions/q/TX/San_Antonio.json'




def loadStation(url):
	f = urllib2.urlopen(url)
	json_string =f.read()
	parsed_json=json.loads(json_string)
	f.close()
	return parsed_json

def parseTemp(parsed_json):
	temp = parsed_json['current_observation']['temp_f']
	return temp

def parseTime(parsed_json):
	time = parsed_json['current_observation']['observation_time']
	return time

def main():
	lasttime = []
	while 0 < 1:
		rep = loadStation(url)
		try:
			temp = parseTemp(rep)
			thime = parseTime(rep)
			if thime != lasttime:
				publish.single("/SanAntonioTemp/Wunder", payload=("At "+str(thime)+" the temp was "+str(temp)),qos=2,hostname="10.0.0.3")
				publish.single("/SanAntonioTemp/Wunder", payload=("At "+str(thime)+" the temp was "+str(temp)),qos=2,hostname="10.3.12.48", port=1883)
				lasttime = thime
				count = 0
				print "At "+str(thime)+ " the temp was "+str(temp)+ " has been published"  
			else:
				count = count + 15
				publish.single("/SanAntonioTemp/Wunder", payload=("Time last updated was "+str(count/60)+ " minutes ago and the temperature was "+str(temp)),qos=2,hostname="10.0.0.3")
				publish.single("/SanAntonioTemp/Wunder", payload=("Time last updated was "+str(count/60)+ " minutes ago and the temperature was "+str(temp)),qos=2,hostname="10.3.12.48", port=1883)
				print "Published new count " +str(count/60)
		except:
			print 'Could not parse data. Will retry in 15 seconds.'
		time.sleep(15)


 

main()

